import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import seed
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, mean_squared_error

seed(42)

# Load price data
price_df = pd.read_csv(r"D:\python\feature engineering bookcamp\ch07\twlo_prices.csv")

# Set datetime index
price_df.index = pd.to_datetime(price_df['date'])
price_df.index = price_df.index.tz_convert('US/Pacific')
price_df.sort_index(inplace=True)
del price_df['date']

last_price_of_the_day = (
    price_df.groupby(price_df.index.date)
    .tail(1)['close']
    .rename('day_close_price')
)

last_price_of_the_day.index = last_price_of_the_day.index.date

price_df['day'] = price_df.index.date
price_df = price_df.merge(last_price_of_the_day, left_on='day', right_index=True)

price_df['pct_change_eod'] = (
    (price_df['day_close_price'] - price_df['close']) 
    / price_df['close']
)

price_df['stock_price_rose'] = price_df['pct_change_eod'] > 0

price_df['feature__dayofweek'] = price_df.index.dayofweek
price_df['feature__morning'] = price_df.index.hour < 12

# Lag Features
price_df['feature__lag_30_min'] = price_df['close'].shift(30, freq='1min')
price_df['feature__lag_7_day'] = price_df['close'].shift(7, freq='D')

# Rolling Features
price_df['feature__rolling_close_mean_60'] = price_df['close'].rolling('60min').mean()
price_df['feature__rolling_close_std_60'] = price_df['close'].rolling('60min').std()
price_df['feature__rolling_volume_mean_60'] = price_df['volume'].rolling('60min').mean()
price_df['feature__rolling_volume_std_60'] = price_df['volume'].rolling('60min').std()

# Expanding Features
price_df['feature__expanding_close_mean'] = price_df['close'].expanding(200).mean()
price_df['feature__expanding_volume_mean'] = price_df['volume'].expanding(200).mean()

price_df.dropna(inplace=True)

clf = RandomForestClassifier(random_state=0)

ml_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('classifier', clf)
])

params = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__min_samples_split': [2, 3, 5],
    'classifier__max_depth': [10, None],
    'classifier__max_features': [None, 'auto']
}

tscv = TimeSeriesSplit(n_splits=2)

def split_data(price_df):
    downsized_price_df = price_df[(price_df.index.minute == 0)]
    
    train_df = downsized_price_df[:'2021-05-31']
    test_df = downsized_price_df['2021-06-01':]

    train_X = train_df.filter(regex='feature')
    test_X = test_df.filter(regex='feature')

    train_y = train_df['stock_price_rose']
    test_y = test_df['stock_price_rose']

    return train_df, test_df, train_X, train_y, test_X, test_y

def advanced_grid_search(x_train, y_train, x_test, y_test,
                         ml_pipeline, params, cv=3,
                         include_probas=False):

    model_grid_search = GridSearchCV(
        ml_pipeline,
        param_grid=params,
        cv=cv,
        error_score=-1
    )

    start_time = time.time()
    model_grid_search.fit(x_train, y_train)

    best_model = model_grid_search.best_estimator_
    y_preds = best_model.predict(x_test)

    print(classification_report(y_test, y_preds))
    print(f'Best params: {model_grid_search.best_params_}')
    print(f"Took {time.time() - start_time:.2f} seconds")

    if include_probas:
        y_probas = best_model.predict_proba(x_test).max(axis=1)
        return best_model, y_preds, y_probas

    return best_model, y_preds

train_df, test_df, train_X, train_y, test_X, test_y = split_data(price_df)

best_model, test_preds, test_probas = advanced_grid_search(
    train_X, train_y,
    test_X, test_y,
    ml_pipeline, params,
    cv=tscv,
    include_probas=True
)

daily_features = pd.DataFrame()

daily_features['first_5_min_avg_close'] = (
    price_df.groupby(price_df.index.date)['close']
    .apply(lambda x: x.head().mean())
)

daily_features['last_5_min_avg_close'] = (
    price_df.groupby(price_df.index.date)['close']
    .apply(lambda x: x.tail().mean())
)

daily_features['feature__overnight_change_close'] = (
    (daily_features['first_5_min_avg_close']
     - daily_features['last_5_min_avg_close'].shift(1))
    / daily_features['last_5_min_avg_close'].shift(1)
)

def macd(ticker):
    exp1 = ticker.ewm(span=12, adjust=False).mean()
    exp2 = ticker.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd.ewm(span=9, adjust=False).mean()

daily_features['feature__macd'] = macd(
    daily_features['first_5_min_avg_close']
)

price_df = price_df.merge(
    daily_features,
    left_on=price_df.index.date,
    right_index=True
)

price_df.dropna(inplace=True)

tweet_df = pd.read_csv(r"D:\python\feature engineering bookcamp\ch07\twlo_tweets.csv", encoding='ISO-8859-1')

tweet_df.index = pd.to_datetime(tweet_df['date_tweeted'],format='mixed',utc=True)
tweet_df.index = tweet_df.index.tz_convert('US/Pacific')
tweet_df.sort_index(inplace=True)
del tweet_df['date_tweeted']

rolling_1_day_verified_count = (
    tweet_df.resample('1min')['author_verified']
    .sum()
    .rolling('1D')
    .sum()
)

rolling_7_day_total_tweets = (
    tweet_df.resample('1min')['tweet_unique_id']
    .count()
    .rolling('7D')
    .sum()
)

twitter_stats = pd.DataFrame({
    'feature__rolling_7_day_total_tweets': rolling_7_day_total_tweets,
    'feature__rolling_1_day_verified_count': rolling_1_day_verified_count
})

price_df = price_df.merge(
    twitter_stats,
    left_index=True,
    right_index=True
)

price_df.dropna(inplace=True)

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# Estimators for feature selection
rf_small = RandomForestClassifier(n_estimators=20, random_state=0)
lr = LogisticRegression(max_iter=1000, random_state=0)

# Pipeline: Polynomial features -> Scaling -> Feature selection -> Classifier
ml_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scale', StandardScaler()),
    ('select_from_model', SelectFromModel(estimator=rf_small)),
    ('classifier', clf)
])

# Grid search parameters
params = {
    'poly__degree': [2],
    'select_from_model__threshold': ['0.5*mean', 'mean', 'median'],
    'select_from_model__estimator': [rf_small, lr],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__min_samples_split': [2, 3, 5],
    'classifier__max_depth': [10, None],
    'classifier__max_features': [None, 'sqrt']
}

train_df, test_df, train_X, train_y, test_X, test_y = split_data(price_df)

# Run time-series cross-validation
best_model, test_preds, test_probas = advanced_grid_search(
    train_X, train_y,
    test_X, test_y,
    ml_pipeline,
    params,
    cv=tscv,
    include_probas=True
)

