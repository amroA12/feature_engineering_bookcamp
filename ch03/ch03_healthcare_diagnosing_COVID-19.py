import pandas as pd
import numpy as np
import time

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from feature_engine.imputation import EndTailImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


covid_flu = pd.read_csv(r'D:\python\feature engineering bookcamp\ch03\covid_flu.csv')


numeric_types = ['float16', 'float32', 'float64', 'int16', 'int32', 'int64']
numerical_columns = covid_flu.select_dtypes(include=numeric_types).columns.tolist()


covid_flu['Age'] = covid_flu['Age'].fillna(covid_flu['Age'].median()) + 0.01


categorical_columns = covid_flu.select_dtypes(include=['O']).columns.tolist()
categorical_columns.remove('Diagnosis')  # الهدف

covid_flu['Female'] = covid_flu['Sex'] == 'F'
del covid_flu['Sex']

covid_flu = covid_flu.replace({'Yes': True, 'No': False})

binary_features = [
    'Female', 'GroundGlassOpacity', 'CTscanResults', 'Diarrhea', 'Fever', 'Coughing', 
    'SoreThroat', 'NauseaVomitting', 'Fatigue', 'InitialPCRDiagnosis'
]

covid_flu['FluSymptoms'] = covid_flu[['Diarrhea','Fever','Coughing','SoreThroat','NauseaVomitting','Fatigue']].sum(axis=1) >= 1
binary_features.append('FluSymptoms')


class DummifyRiskFactor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_binarizer = None
        
    def parse_risk_factors(self, comma_sep_factors):
        try:
            return [s.strip().lower() for s in comma_sep_factors.split(',')]
        except:
            return []
    
    def fit(self, X, y=None):
        self.label_binarizer = MultiLabelBinarizer()
        self.label_binarizer.fit(X.apply(self.parse_risk_factors))
        return self
    
    def transform(self, X, y=None):
        return self.label_binarizer.transform(X.apply(self.parse_risk_factors))


X, y = covid_flu.drop(['Diagnosis'], axis=1), covid_flu['Diagnosis']
x_train, x_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, test_size=0.2
)


risk_factor_pipeline = Pipeline([
    ('select_risk_factor', FunctionTransformer(lambda df: df['RiskFactors'])),
    ('fillna', FunctionTransformer(lambda s: s.fillna(''))),
    ('dummify', DummifyRiskFactor()),
    ('select_kbest', SelectKBest(mutual_info_classif, k=20))
])

binary_pipeline = Pipeline([
    ('select_binary', FunctionTransformer(lambda df: df[binary_features])),
    ('impute', SimpleImputer(strategy='constant', fill_value=False))
])

numerical_pipeline = Pipeline([
    ('select_numeric', FunctionTransformer(lambda df: df[numerical_columns])),
    ('impute', SimpleImputer(strategy='median')),
    ('boxcox', PowerTransformer(method='box-cox', standardize=True)),
    ('endtail', EndTailImputer(imputation_method='gaussian')),
    ('scale', StandardScaler()),
    ('bins', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans'))
])

simple_fe = FeatureUnion([
    ('risk_factors', risk_factor_pipeline),
    ('binary_pipeline', binary_pipeline),
    ('numerical_pipeline', numerical_pipeline)
])


def simple_grid_search(x_train, y_train, x_test, y_test, feature_engineering_pipeline=None):
    params = {
        'max_depth': [10, None],
        'n_estimators': [50, 100],
        'criterion': ['gini', 'entropy']
    }
    base_model = ExtraTreesClassifier(random_state=0)
    model_grid_search = GridSearchCV(base_model, param_grid=params, cv=3)
    
    start_time = time.time()
    
    if feature_engineering_pipeline:
        parsed_x_train = feature_engineering_pipeline.fit_transform(x_train, y_train)
        parsed_x_test = feature_engineering_pipeline.transform(x_test)
    else:
        parsed_x_train = x_train
        parsed_x_test = x_test
    
    parse_time = time.time()
    print(f"Feature engineering took {(parse_time - start_time):.2f} seconds")
    
    model_grid_search.fit(parsed_x_train, y_train)
    fit_time = time.time()
    print(f"Training took {(fit_time - parse_time):.2f} seconds")
    
    best_model = model_grid_search.best_estimator_
    
    print(classification_report(y_test, best_model.predict(parsed_x_test)))
    end_time = time.time()
    print(f"Overall took {(end_time - start_time):.2f} seconds\n")
    
    return best_model


best_model = simple_grid_search(x_train, y_train, x_test, y_test, simple_fe)

