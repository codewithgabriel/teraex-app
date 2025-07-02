# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 04:34:56 2025

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN


# Load dataset
df = pd.read_csv("BTCUSDTM15ac.csv", header=None, encoding='utf-16', sep=',')
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
target = ['close']
df = df.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32', 'marketCap': 'float32'})

df= df[["open" , "high", "close", "low", "volume", "close"]]


#%%

def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    
    
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i+sequence_length].values)
        labels.append(data[target].values[i+sequence_length])
     
        
    return np.array(sequences, dtype='float32'), np.array(labels , dtype='float32')

sequence_length = 60  # Use 60 15 mins candles of data to predict the next day
X, y = create_sequences(df, sequence_length)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
#%%


rnn_model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])

rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(X_train, y_train, epochs=5, batch_size=32)

#%%


lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)


#%%
#%% Create future timestamps
from datetime import timedelta
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








future_X_raw = future_df[features]
future_X_scaled = scaler.transform(future_X_raw)

#%%
# Predict using RNN
rnn_predictions = rnn_model.predict(X_test)
rnn_predictions = scaler.inverse_transform(rnn_predictions)

# Predict using LSTM
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='True Prices')
plt.plot(df.index[-len(y_test):], rnn_predictions, label='RNN Predictions')
#plt.plot(df.index[-len(y_test):], lstm_predictions, label='LSTM Predictions')
plt.legend()
plt.show()
