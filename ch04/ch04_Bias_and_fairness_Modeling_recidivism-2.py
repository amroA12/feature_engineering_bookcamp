# ===============================
# Import Libraries
# ===============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import dalex as dx

# ===============================
# Data Loading
# ===============================
compas_df = pd.read_csv(r"D:\python\feature engineering bookcamp\ch04\compas-scores-two-years.csv")

# Select relevant columns
compas_df = compas_df[["sex", "age", "race", "juv_fel_count", "juv_misd_count",
                       "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]]

# Combine juvenile counts
compas_df['juv_count'] = compas_df[["juv_fel_count", "juv_misd_count", "juv_other_count"]].sum(axis=1)
compas_df = compas_df.drop(["juv_fel_count", "juv_misd_count", "juv_other_count"], axis=1)

# Train-test split
X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
    compas_df.drop('two_year_recid', axis=1),
    compas_df['two_year_recid'],
    compas_df['race'],
    stratify=compas_df['two_year_recid'],
    test_size=0.3,
    random_state=0
)

# ===============================
# DIR Transformer
# ===============================
class NormalizeColumnByLabel(BaseEstimator, TransformerMixin):
    def __init__(self, col, label):
        self.col = col
        self.label = label
        self.transformers = {}
        
    def fit(self, X, y=None):
        for group in X[self.label].unique():
            self.transformers[group] = PowerTransformer(method='yeo-johnson', standardize=True)
            self.transformers[group].fit(X.loc[X[self.label]==group][self.col].values.reshape(-1, 1))
        return self
    
    def transform(self, X, y=None):
        C = X.copy()
        for group in X[self.label].unique():
            C.loc[X[self.label]==group, self.col] = self.transformers[group].transform(
                X.loc[X[self.label]==group][self.col].values.reshape(-1, 1)
            )
        return C

# ===============================
# Preprocessing
# ===============================
categorical_features = ['race', 'sex', 'c_charge_degree']
numerical_features = ["age", "priors_count"]

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='if_binary'))
])

numerical_transformer = Pipeline([
    ('scale', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, numerical_features)
])

# ===============================
# Model: DIR Random Forest
# ===============================
classifier = RandomForestClassifier(max_depth=10, n_estimators=20, random_state=0)

clf_tree_aware = Pipeline([
    ('normalize_priors', NormalizeColumnByLabel(col='priors_count', label='race')),
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

clf_tree_aware.fit(X_train, y_train)
y_preds_aware = clf_tree_aware.predict(X_test)

# ===============================
# Performance
# ===============================
print("DIR MODEL (Disparate Impact Remover)")
print("="*50)
print(classification_report(y_test, y_preds_aware))

# DALEX explanation
exp_tree_aware = dx.Explainer(clf_tree_aware, X_test, y_test, label='Random Forest DIR', verbose=False)
print(exp_tree_aware.model_performance())

