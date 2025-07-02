# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 06:35:55 2025

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

#%% Load MT5 CSV (no headers)
df = pd.read_csv("ETHUSDTM30.csv", header=None, encoding='utf-16' , sep=',')

df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ignore']
df = df.drop(columns=['ignore'])
print(df.head())
print(df.shape)

df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y.%m.%d %H:%M')

# Optional: If needed, convert volume to float
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

# Sort just in case
df = df.sort_values(by='timestamp')

# Dummy marketCap (not in MT5), fill with rolling sum of volume as proxy
df['marketCap'] = df['volume'].rolling(window=60, min_periods=1).sum()

#%% Feature engineering
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
    "Ridge": Ridge(alpha=1.0),
}
for model in models.values():
    model.fit(X_train, y_train)

#%% Forecast next 10 minutes (2 candles ahead at 5-minute intervals)
# Use last 10 rows to fit a linear trend for price-related features
n_history = 10
history = df.tail(n_history)

# Fit linear trend (degree 1 polynomial) for each feature
trend_features = {}
for feature in ['open', 'high', 'low', 'volume', 'marketCap']:
    trend = np.polyfit(range(n_history), history[feature].values, deg=1)
    trend_features[feature] = trend  # (slope, intercept)

# Generate future timestamps (10 steps of 5 minutes = next 50 minutes)
num_steps = 10
interval = timedelta(minutes=30)
future_times = [df['timestamp'].iloc[-1] + i * interval for i in range(1, num_steps + 1)]

# Build future feature rows with trend-based estimates
future_features = []
for i, time in enumerate(future_times, start=1):
    row = {
        'open': trend_features['open'][0] * (n_history + i) + trend_features['open'][1],
        'high': trend_features['high'][0] * (n_history + i) + trend_features['high'][1],
        'low': trend_features['low'][0] * (n_history + i) + trend_features['low'][1],
        'volume': trend_features['volume'][0] * (n_history + i) + trend_features['volume'][1],
        'marketCap': trend_features['marketCap'][0] * (n_history + i) + trend_features['marketCap'][1],
        'day_of_week': time.dayofweek,
        'month': time.month,
        'day': time.day
    }
    future_features.append(row)

future_df = pd.DataFrame(future_features)


#%%
# Step 1: Get predictions from the Random Forest model
rf_model = models['RandomForest']
rf_preds = rf_model.predict(future_df)

gb_model = models["GradientBoosting"]
gb_preds = gb_model.predict(future_df)

rg_model = models["Ridge"]
rg_preds = rg_model.predict(future_df)

# Combining their predictions 

all_preds = np.stack([rf_preds , gb_preds, rg_preds] , axis=1)
average_all_preds = np.mean(all_preds, axis=1)



plt.figure(figsize=(12, 5))
plt.plot(future_times, rf_preds, color='green', marker='o', linestyle='-', label='RF Forecast')
plt.plot(future_times, gb_preds, color='blue', marker='o', linestyle='-', label='GB Forecast')
plt.plot(future_times, rg_preds, color='orange', marker='o', linestyle='-', label='Ridges Forecast')
plt.plot(future_times, average_all_preds, color='red', marker='o', linestyle='-', label='Average Forecast')
plt.title(f'Random Forest Forecast (Next 10 Candles at 15-min Intervals)')
plt.xlabel('Time')
plt.ylabel('Predicted Close Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ridge_confidence_interval(model, X, y, X_future, alpha=0.05):
    from scipy.stats import t
    preds = model.predict(X_future)
    y_pred = model.predict(X)
    residuals = y - y_pred
    dof = len(y) - X.shape[1]
    std_err = np.sqrt(np.sum(residuals**2) / dof)
    
    t_score = t.ppf(1 - alpha/2, dof)
    ci = t_score * std_err
    lower = preds - ci
    upper = preds + ci
    return lower, upper

def bootstrap_ci(model, X_train, y_train, X_future, n_iterations=100, alpha=0.05):
    preds = []
    for _ in range(n_iterations):
        X_resampled, y_resampled = resample(X_train, y_train)
        model.fit(X_resampled, y_resampled)
        preds.append(model.predict(X_future))
    preds = np.array(preds)
    lower = np.percentile(preds, 100 * alpha/2.0, axis=0)
    upper = np.percentile(preds, 100 * (1 - alpha/2.0), axis=0)
    return lower, upper


def evaluate_model(model, X, y, model_name=''):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1)/(n - p - 1)
    
    print(f"\n📊 Metrics for {model_name}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2
    }


#%% Assuming you're using the same X and y for training


# Confidence Intervals
rf_lower, rf_upper = bootstrap_ci(rf_model, X_train, y_train, future_df)
gb_lower, gb_upper = bootstrap_ci(gb_model, X_train, y_train, future_df)
rg_lower, rg_upper = ridge_confidence_interval(rg_model, X_train, y_train, future_df)

#%% Evaluation Metrics
rf_metrics = evaluate_model(rf_model, X_train, y_train, "Random Forest")
gb_metrics = evaluate_model(gb_model, X_train, y_train, "Gradient Boosting")
rg_metrics = evaluate_model(rg_model, X_train, y_train, "Ridge Regression")



rf_ci_width = np.mean(rf_upper - rf_lower)
gb_ci_width = np.mean(gb_upper - gb_lower)
rg_ci_width = np.mean(rg_upper - rg_lower)
avg_ci_width = np.mean([rf_ci_width, gb_ci_width, rg_ci_width])

print("\n📏 Average Confidence Interval Widths:")
print(f"Random Forest CI Avg Width: {rf_ci_width:.4f}")
print(f"Gradient Boosting CI Avg Width: {gb_ci_width:.4f}")
print(f"Ridge Regression CI Avg Width: {rg_ci_width:.4f}")
print(f"Overall Avg CI Width: {avg_ci_width:.4f}")

#%% Predict using models
predictions_all = {}
ci_all = {}

for name, model in models.items():
    preds = model.predict(future_df)
    if name == "RandomForest":
        stds = np.std([tree.predict(future_df) for tree in model.estimators_], axis=0)
    else:
        stds = np.full_like(preds, np.std(np.abs(y_test - model.predict(X_test))))

    ci_lower = preds - 1.96 * stds
    ci_upper = preds + 1.96 * stds

    predictions_all[name] = preds
    ci_all[name] = (ci_lower, ci_upper)

#%% Average prediction
avg_preds = np.mean(np.column_stack(list(predictions_all.values())), axis=1)
avg_ci_lower = np.mean(np.column_stack([ci_all[m][0] for m in models]), axis=1)
avg_ci_upper = np.mean(np.column_stack([ci_all[m][1] for m in models]), axis=1)

#%% Plot
plt.figure(figsize=(14, 6))
plt.plot(future_times, avg_preds, label='Average Prediction', color='blue')
plt.fill_between(future_times, avg_ci_lower, avg_ci_upper, color='blue', alpha=0.2, label='95% CI')
plt.xlabel('Time (Next 2 Hours - 1 min interval)')
plt.ylabel('Predicted Close Price')
plt.title('Price Forecast (Next 2 Hours, 1-Minute Interval)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




