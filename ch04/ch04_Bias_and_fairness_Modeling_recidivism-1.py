# ===============================
# Import Libraries
# ===============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import dalex as dx

# ===============================
# Load Data
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
# Model: Baseline Random Forest
# ===============================
classifier = RandomForestClassifier(max_depth=10, n_estimators=20, random_state=0)

clf_tree = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

clf_tree.fit(X_train, y_train)
y_preds = clf_tree.predict(X_test)

# ===============================
# Performance
# ===============================
print("BASELINE MODEL (FAIRNESS UNAWARE)")
print("="*50)
print(classification_report(y_test, y_preds))

# DALEX explanation
exp_tree = dx.Explainer(clf_tree, X_test, y_test, label='Random Forest Unaware', verbose=False)
print(exp_tree.model_performance())

