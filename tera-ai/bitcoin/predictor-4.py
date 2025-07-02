# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 01:42:26 2025

@author: HP
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

#%%
# Re-load the file and rerun model training and evaluation

# Load the dataset again
file_path = "Bitcoin_4_4_2024-4_4_2025_historical_data_coinmarketcap.csv"
df = pd.read_csv(file_path, sep=';')

# Clean and prepare timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('"', ''), format='ISO8601')
df = df.sort_values(by='timestamp')

# Feature engineering
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day

features = ['open', 'high', 'low', 'volume', 'marketCap', 'day_of_week', 'month', 'day']
target = 'close'

# Prepare training and testing data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#%% Define three different regression models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
    "Ridge": Ridge(alpha=1.0)
}

# Train models and collect predictions
predictions = {}
intervals = {}

# Use 2 days ahead timestamp for consistency
last_row = df.iloc[-1]
future_time = last_row['timestamp'] + timedelta(days=2)

next_day_features = {
    'open': last_row['open'],
    'high': last_row['high'],
    'low': last_row['low'],
    'volume': last_row['volume'],
    'marketCap': last_row['marketCap'],
    'day_of_week': future_time.dayofweek,
    'month': future_time.month,
    'day': future_time.day
}
next_day_df = pd.DataFrame([next_day_features])

for name, model in models.items():
    model.fit(X_train, y_train)
    if name == "RandomForest":
        tree_preds = np.array([tree.predict(next_day_df)[0] for tree in model.estimators_])
        mean_pred = np.mean(tree_preds)
        std_pred = np.std(tree_preds)
    elif name == "GradientBoosting":
        preds = np.array([model.predict(next_day_df)[0] for _ in range(100)])
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
    else:  # Ridge or any single output model
        mean_pred = model.predict(next_day_df)[0]
        std_pred = np.std(np.abs(y_test - model.predict(X_test)))  # simple residual-based std estimate
    
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    
    predictions[name] = mean_pred
    intervals[name] = (ci_lower, ci_upper)

#%% Average prediction and average confidence interval
avg_prediction = np.mean(list(predictions.values()))
avg_ci_lower = np.mean([ci[0] for ci in intervals.values()])
avg_ci_upper = np.mean([ci[1] for ci in intervals.values()])

(predictions, intervals, avg_prediction, (avg_ci_lower, avg_ci_upper))
