# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 01:58:11 2025

@author: HP
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta

#%% Reload data
df = pd.read_csv("Bitcoin_4_6_2024-4_6_2025_historical_data_coinmarketcap.csv", sep=';')
df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('"', ''), format='%Y-%m-%dT%H:%M:%S.%fZ')
df = df.sort_values(by='timestamp')

# Feature engineering
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
features = ['open', 'high', 'low', 'volume', 'marketCap', 'day_of_week', 'month', 'day']
target = 'close'

#%% Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#%% Define and train models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
    "Ridge": Ridge(alpha=1.0)
}
for model in models.values():
    model.fit(X_train, y_train)

#%% Create future timestamps for every minute in the next 2 hours
last_row = df.iloc[-1]
start_time = last_row['timestamp'] + timedelta(hours=24)
future_times = [start_time + timedelta(minutes=i) for i in range(120)]

# Generate future feature rows
future_features = []
for time in future_times:
    row = {
        'open': last_row['open'],
        'high': last_row['high'],
        'low': last_row['low'],
        'volume': last_row['volume'],
        'marketCap': last_row['marketCap'],
        'day_of_week': time.dayofweek,
        'month': time.month,
        'day': time.day
    }
    future_features.append(row)
future_df = pd.DataFrame(future_features)

#%% Predict using all models for each future timestamp
predictions_all = {}
ci_all = {}

for name, model in models.items():
    preds = model.predict(future_df)
    if name == "RandomForest":
        stds = np.std([tree.predict(future_df) for tree in model.estimators_], axis=0)
    elif name == "GradientBoosting":
        stds = np.full_like(preds, np.std(np.abs(y_test - model.predict(X_test))))
    else:
        stds = np.full_like(preds, np.std(np.abs(y_test - model.predict(X_test))))

    ci_lower = preds - 1.96 * stds
    ci_upper = preds + 1.96 * stds

    predictions_all[name] = preds
    ci_all[name] = (ci_lower, ci_upper)

#%% Average predictions
avg_preds = np.mean(np.column_stack(list(predictions_all.values())), axis=1)
avg_ci_lower = np.mean(np.column_stack([ci_all[m][0] for m in models]), axis=1)
avg_ci_upper = np.mean(np.column_stack([ci_all[m][1] for m in models]), axis=1)

#%% Plot results
plt.figure(figsize=(14, 6))
plt.plot(future_times, avg_preds, label='Average Prediction', color='blue')
plt.fill_between(future_times, avg_ci_lower, avg_ci_upper, color='blue', alpha=0.2, label='Average 95% CI')
plt.xlabel('Time (Next 2 Hours - 1 min interval)')
plt.ylabel('Predicted BTC Close Price')
plt.title('BTC Price Forecast (Next 2 Hours, 1-Minute Interval)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Output predictions for 1-day ahead from each model
future_1_day = last_row['timestamp'] + timedelta(days=1)
single_day_features = pd.DataFrame([{
    'open': last_row['open'],
    'high': last_row['high'],
    'low': last_row['low'],
    'volume': last_row['volume'],
    'marketCap': last_row['marketCap'],
    'day_of_week': future_1_day.dayofweek,
    'month': future_1_day.month,
    'day': future_1_day.day
}])

single_day_preds = {}
single_day_ci = {}

for name, model in models.items():
    pred = model.predict(single_day_features)[0]
    if name == "RandomForest":
        std = np.std([tree.predict(single_day_features)[0] for tree in model.estimators_])
    else:
        std = np.std(np.abs(y_test - model.predict(X_test)))
    lower = pred - 1.96 * std
    upper = pred + 1.96 * std
    single_day_preds[name] = pred
    single_day_ci[name] = (lower, upper)

# Average of 1-day prediction and CI
avg_1day_pred = np.mean(list(single_day_preds.values()))
avg_1day_ci_lower = np.mean([ci[0] for ci in single_day_ci.values()])
avg_1day_ci_upper = np.mean([ci[1] for ci in single_day_ci.values()])

(single_day_preds, single_day_ci, avg_1day_pred, (avg_1day_ci_lower, avg_1day_ci_upper))

#%% outputing results for clarity. 
rf_pred = single_day_preds['RandomForest']
gb_pred = single_day_preds['GradientBoosting']
rd_pred = single_day_preds['Ridge']

rf_ci = single_day_ci['RandomForest']
gb_ci = single_day_ci['GradientBoosting']
rd_ci = single_day_ci['Ridge']

average_pred = np.mean([rf_pred , gb_pred , rd_pred])


print("*" * 50)
print(f"Random Forest for {future_1_day.date()} is: {rf_pred} ")
print(f"Gradient Boosting for {future_1_day.date()} is: {gb_pred} ")
print(f"Ridge for {future_1_day.date()} is: {rd_pred} ")
print(f"Average prediction is: {average_pred}")

print("*" * 50)
print(f"Random Forest CI is: {rf_ci} ")
print(f"Gradient Boosting CI  is: {gb_ci} ")
print(f"Ridge CI is: {rd_ci} ")
print("*" * 50)
print(f"Average Lower Bound Confidence Interval is: {avg_1day_ci_lower}")
print(f"Average Upper Bound Confidence Interval is: {avg_1day_ci_upper}")

