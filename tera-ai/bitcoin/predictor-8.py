

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 06:35:55 2025
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.utils import resample
from scipy.stats import t

#%% Load and prepare dataset
df = pd.read_csv("ETHUSDTM30.csv", header=None, encoding='utf-16', sep=',')
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ignore']
df.drop(columns=['ignore'], inplace=True)

df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y.%m.%d %H:%M')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
df = df.sort_values('timestamp')

# Proxy marketCap
df['marketCap'] = df['volume'].rolling(window=60, min_periods=1).sum()

# Features
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
features = ['open', 'high', 'low', 'volume', 'marketCap', 'day_of_week', 'month', 'day']
target = 'close'

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

#%% Trend-based future feature creation
n_history = 10
history = df.tail(n_history)
trend_features = {}
for feature in ['open', 'high', 'low', 'volume', 'marketCap']:
    trend_features[feature] = np.polyfit(range(n_history), history[feature].values, deg=1)

num_steps = 10
interval = timedelta(minutes=30)
future_times = [df['timestamp'].iloc[-1] + i * interval for i in range(1, num_steps + 1)]

future_features = []
for i, time in enumerate(future_times, start=1):
    row = {feat: trend_features[feat][0] * (n_history + i) + trend_features[feat][1] for feat in trend_features}
    row.update({'day_of_week': time.dayofweek, 'month': time.month, 'day': time.day})
    future_features.append(row)

future_df = pd.DataFrame(future_features)

#%% Predict
predictions_all = {}
ci_all = {}

for name, model in models.items():
    preds = model.predict(future_df)
    
    # Estimate standard deviation for CI
    if name == "RandomForest":
        stds = np.std([tree.predict(future_df) for tree in model.estimators_], axis=0)
    else:
        residuals = y_test - model.predict(X_test)
        stds = np.full_like(preds, np.std(residuals))
    
    ci_lower = preds - 1.96 * stds
    ci_upper = preds + 1.96 * stds

    predictions_all[name] = preds
    ci_all[name] = (ci_lower, ci_upper)

#%% Average prediction + CI
avg_preds = np.mean(np.column_stack(list(predictions_all.values())), axis=1)
avg_ci_lower = np.mean(np.column_stack([ci_all[m][0] for m in models]), axis=1)
avg_ci_upper = np.mean(np.column_stack([ci_all[m][1] for m in models]), axis=1)

#%% Plot
plt.figure(figsize=(14, 6))
for name, preds in predictions_all.items():
    plt.plot(future_times, preds, marker='o', linestyle='--', label=f'{name} Forecast')

plt.plot(future_times, avg_preds, label='Average Forecast', color='red', linewidth=2)
plt.fill_between(future_times, avg_ci_lower, avg_ci_upper, color='red', alpha=0.2, label='95% CI (Avg)')
plt.xlabel('Time')
plt.ylabel('Predicted Close Price')
plt.title('Forecast (Next 5 Hours, 30-Min Intervals)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Evaluation metrics
def evaluate_model(model, X, y, model_name=''):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    n, p = X.shape
    adj_r2 = 1 - (1 - r2) * (n - 1)/(n - p - 1)
    
    print(f"\n📊 Metrics for {model_name}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Adjusted R²: {adj_r2:.4f}")
    
    return mse, rmse, mae, r2, adj_r2

for name, model in models.items():
    evaluate_model(model, X_test, y_test, name)

#%% Confidence Interval widths
ci_widths = {name: np.mean(upper - lower) for name, (lower, upper) in ci_all.items()}
avg_ci_width = np.mean(list(ci_widths.values()))

print("\n📏 Average Confidence Interval Widths:")
for name, width in ci_widths.items():
    print(f"  {name}: {width:.4f}")
print(f"  Overall Avg CI Width: {avg_ci_width:.4f}")


#%% 

from sklearn.preprocessing import StandardScaler
import ta  # Technical Analysis library (if not installed, use: pip install ta)

# Make a copy to avoid mutating original data
df_feat = df.copy()

# === Feature Engineering ===

# Technical indicators from 'ta' library
df_feat['rsi'] = ta.momentum.RSIIndicator(close=df_feat['close']).rsi()
df_feat['macd'] = ta.trend.MACD(close=df_feat['close']).macd()
df_feat['macd_signal'] = ta.trend.MACD(close=df_feat['close']).macd_signal()
df_feat['ema_10'] = ta.trend.EMAIndicator(close=df_feat['close'], window=10).ema_indicator()
df_feat['ema_30'] = ta.trend.EMAIndicator(close=df_feat['close'], window=30).ema_indicator()
df_feat['volatility'] = ta.volatility.BollingerBands(close=df_feat['close']).bollinger_hband() - \
                        ta.volatility.BollingerBands(close=df_feat['close']).bollinger_lband()

# Lag features (previous time step values)
for lag in range(1, 4):
    df_feat[f'close_lag_{lag}'] = df_feat['close'].shift(lag)
    df_feat[f'volume_lag_{lag}'] = df_feat['volume'].shift(lag)

# Drop rows with NaNs from indicators/lagging
df_feat.dropna(inplace=True)

# Define features and target
features = [
    'open', 'high', 'low', 'volume', 'rsi', 'macd', 'macd_signal',
    'ema_10', 'ema_30', 'volatility',
    'close_lag_1', 'close_lag_2', 'close_lag_3',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3'
]
target = 'close'

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat[features])
y = df_feat[target]

# Train/test split without shuffling (time series aware)
X_train, X_test = X_scaled[:-2000], X_scaled[-2000:]
y_train, y_test = y[:-2000], y[-2000:]

# Return feature set shape and preview
X_train.shape, y_train.shape, df_feat[features + [target]].head()


#%%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Ridge": Ridge(alpha=1.0),
}

# Train all models
for model in models.values():
    model.fit(X_train, y_train)

#%% Evaluation function
def evaluate_model(model, X, y, model_name=''):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1)/(n - p - 1)
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2
    }

# Evaluate all models on test set
results = []
for name, model in models.items():
    metrics = evaluate_model(model, X_test, y_test, model_name=name)
    results.append(metrics)

results
