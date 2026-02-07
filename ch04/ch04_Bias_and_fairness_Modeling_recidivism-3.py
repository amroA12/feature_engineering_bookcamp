# ===============================
# Imports
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing.lfr import LFR

import dalex as dx


# ===============================
# Data Loading & Selection
# ===============================
compas_df = pd.read_csv(
    r"D:\python\feature engineering bookcamp\ch04\compas-scores-two-years.csv"
)

compas_df = compas_df[
    [
        "sex", "age", "race",
        "juv_fel_count", "juv_misd_count", "juv_other_count",
        "priors_count", "c_charge_degree",
        "two_year_recid"
    ]
]

# Combine juvenile counts
compas_df["juv_count"] = compas_df[
    ["juv_fel_count", "juv_misd_count", "juv_other_count"]
].sum(axis=1)

compas_df.drop(
    ["juv_fel_count", "juv_misd_count", "juv_other_count"],
    axis=1,
    inplace=True
)


# ===============================
# Train / Test Split
# ===============================
X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
    compas_df.drop("two_year_recid", axis=1),
    compas_df["two_year_recid"],
    compas_df["race"],
    stratify=compas_df["two_year_recid"],
    test_size=0.3,
    random_state=0
)


# ===============================
# Categorical Encoding
# ===============================
def encode_categorical_data(df):
    """Encode categorical features into numeric values."""
    df = df.copy()

    race_mapping = {
        "Caucasian": 0,
        "African-American": 1,
        "Hispanic": 2,
        "Other": 3
    }
    sex_mapping = {"Male": 0, "Female": 1}
    charge_mapping = {"M": 0, "F": 1}

    df["race"] = df["race"].apply(lambda x: race_mapping.get(x, 3))
    df["sex"] = df["sex"].apply(lambda x: sex_mapping.get(x, 0))
    df["c_charge_degree"] = df["c_charge_degree"].apply(
        lambda x: charge_mapping.get(x, 0)
    )

    return df.dropna()


X_train_encoded = encode_categorical_data(X_train)
X_test_encoded = encode_categorical_data(X_test)


# ===============================
# DIR Transformer
# ===============================
class NormalizeColumnByLabel(BaseEstimator, TransformerMixin):
    """Apply PowerTransformer per protected group."""

    def __init__(self, col_index, label_index):
        self.col_index = col_index
        self.label_index = label_index
        self.transformers = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)

        for group in X_df.iloc[:, self.label_index].unique():
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            mask = X_df.iloc[:, self.label_index] == group
            pt.fit(X_df.loc[mask, self.col_index].values.reshape(-1, 1))
            self.transformers[group] = pt

        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        for group, transformer in self.transformers.items():
            mask = X_df.iloc[:, self.label_index] == group
            X_df.loc[mask, self.col_index] = transformer.transform(
                X_df.loc[mask, self.col_index].values.reshape(-1, 1)
            ).flatten()

        return X_df.values


# ===============================
# LFR Transformer
# ===============================
class LFRCustom(BaseEstimator, TransformerMixin):
    """Learning Fair Representations (AIF360)."""

    def __init__(self, race_index, target_name="response"):
        self.race_index = race_index
        self.target_name = target_name
        self.TR = None

    def fit(self, X, y):
        df = pd.DataFrame(X)
        df[self.target_name] = y

        privileged = [{str(self.race_index): 0.0}]
        unprivileged = [{str(self.race_index): 1.0}]

        binary_df = BinaryLabelDataset(
            df=df,
            protected_attribute_names=[str(self.race_index)],
            label_names=[self.target_name]
        )

        self.TR = LFR(
            unprivileged_groups=unprivileged,
            privileged_groups=privileged,
            k=5, Ax=0.5, Ay=0.2, Az=0.2,
            seed=0, verbose=1
        )

        self.TR.fit(binary_df, maxiter=5000, maxfun=5000)
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X)
        df[self.target_name] = 0

        binary_df = BinaryLabelDataset(
            df=df,
            protected_attribute_names=[str(self.race_index)],
            label_names=[self.target_name]
        )

        transformed = self.TR.transform(binary_df).convert_to_dataframe()[0]
        return transformed.drop(self.target_name, axis=1).values


# ===============================
# Complete Preprocessing Pipeline
# ===============================
priors_index = 3
race_index = 2


class CompletePipeline(BaseEstimator, TransformerMixin):
    """DIR + LFR + Scaling"""

    def __init__(self):
        self.dir = NormalizeColumnByLabel(priors_index, race_index)
        self.lfr = LFRCustom(race_index)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_dir = self.dir.fit_transform(X, y)
        self.lfr.fit(X_dir, y)
        X_lfr = self.lfr.transform(X_dir)
        self.scaler.fit(X_lfr)
        return self

    def transform(self, X, y=None):
        X_dir = self.dir.transform(X)
        X_lfr = self.lfr.transform(X_dir)
        return self.scaler.transform(X_lfr)


# ===============================
# Model Pipeline
# ===============================
model = Pipeline([
    ("preprocessing", CompletePipeline()),
    ("classifier", RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        random_state=0
    ))
])

model.fit(X_train_encoded.values, y_train.values)

y_pred = model.predict(X_test_encoded.values)


# ===============================
# Evaluation
# ===============================
print(classification_report(y_test, y_pred))


# ===============================
# Model Explanation (DALEX)
# ===============================
try:
    explainer = dx.Explainer(
        model,
        X_test_encoded.values,
        y_test,
        label="RF + DIR + LFR",
        verbose=False
    )
    print(explainer.model_performance())
except Exception as e:
    print("DALEX error:", e)

