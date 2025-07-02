# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 03:05:42 2025

@author: HP
"""



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
df = pd.read_csv("BTCUSDTM1.csv", header=None, encoding='utf-16', sep=',')
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

#%% Scale features
from sklearn.decomposition import PCA

# Scale first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat[features])
y = df_feat[target]
# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

#% Split
from sklearn.model_selection import train_test_split 


X_train , X_test , y_train , y_test = train_test_split( X_pca, y , test_size=0.2)
#X_train, X_test = X_pca[:-2000], X_pca[-2000:]
#y_train, y_test = y[:-2000], y[-2000:]


#%% Train the model with train set

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


#%% Create future timestamps
n_history = 10
num_steps = 10
interval = timedelta(minutes=15)

future_times = [df['timestamp'].iloc[-1] + i * interval for i in range(1, num_steps + 1)]

# Use trend lines for base features
trend_features = {}
for feature in ['open', 'high', 'low', 'volume', 'marketCap']:
    trend_features[feature] = np.polyfit(range(n_history), df[feature].tail(n_history).values, deg=1)

# Build future_df
future_features = []
for i, time in enumerate(future_times, start=1):
    row = {feat: trend_features[feat][0] * (n_history + i) + trend_features[feat][1] for feat in trend_features}
    row.update({'day_of_week': time.dayofweek, 'month': time.month, 'day': time.day})
    future_features.append(row)

future_df = pd.DataFrame(future_features)



#Add static indicators using the latest available values
last_close = df['close'].iloc[-1]
last_volume = df['volume'].iloc[-1]

future_df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]
future_df['macd'] = ta.trend.MACD(close=df['close']).macd().iloc[-1]
future_df['macd_signal'] = ta.trend.MACD(close=df['close']).macd_signal().iloc[-1]
future_df['ema_10'] = ta.trend.EMAIndicator(close=df['close'], window=10).ema_indicator().iloc[-1]
future_df['ema_30'] = ta.trend.EMAIndicator(close=df['close'], window=30).ema_indicator().iloc[-1]
bb = ta.volatility.BollingerBands(close=df['close'])
future_df['volatility'] = bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]

for lag in range(1, 4):
    future_df[f'close_lag_{lag}'] = df['close'].iloc[-lag]
    future_df[f'volume_lag_{lag}'] = df['volume'].iloc[-lag]




future_X_raw = future_df[features]
future_X_scaled = scaler.transform(future_X_raw)
future_X_pca = pca.transform(future_X_scaled)

#%% Make predictions with CI
predictions_all = {}
ci_all = {}

for name, model in models.items():
    preds = model.predict(future_X_pca)
    
    if name == "RandomForest":
        stds = np.std([tree.predict(future_X_pca) for tree in model.estimators_], axis=0)
    else:
        residuals = y_test - model.predict(X_test)
        stds = np.full_like(preds, np.std(residuals))

    ci_lower = preds - 1.96 * stds
    ci_upper = preds + 1.96 * stds

    predictions_all[name] = preds
    ci_all[name] = (ci_lower, ci_upper)




#%%
avg_preds = np.mean(np.column_stack(list(predictions_all.values())), axis=1)
avg_ci_lower = np.mean(np.column_stack([ci_all[m][0] for m in models]), axis=1)
avg_ci_upper = np.mean(np.column_stack([ci_all[m][1] for m in models]), axis=1)



rf_model = models['RandomForest']
rf_preds = rf_model.predict(future_X_pca)

gb_model = models["GradientBoosting"]
gb_preds = gb_model.predict(future_X_pca)

rg_model = models["Ridge"]
rg_preds = rg_model.predict(future_X_pca)

# Combining their predictions 

all_preds = np.stack([rf_preds , gb_preds, rg_preds] , axis=1)
average_all_preds = np.mean(all_preds, axis=1)



plt.figure(figsize=(12, 5))
plt.plot(future_times, rf_preds, color='green', marker='o', linestyle='-', label='RF Forecast')
plt.plot(future_times, gb_preds, color='blue', marker='o', linestyle='-', label='GB Forecast')
plt.plot(future_times, rg_preds, color='orange', marker='o', linestyle='-', label='Ridges Forecast')
plt.plot(future_times, average_all_preds, color='red', marker='o', linestyle='-', label='Average Forecast')
plt.fill_between(future_times, avg_ci_lower, avg_ci_upper, color='red', alpha=0.2, label='95% CI (Avg)')
plt.title('Random Forest Forecast (Next 10 Candles at 15-min Intervals)')
plt.xlabel('Time')
plt.ylabel('Predicted Close Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
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

for name, model in models.items():
    evaluate_model(model, X_test, y_test, name)


