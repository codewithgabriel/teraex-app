# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:05:12 2025

@author: HP
"""

#%%
import pandas as pd

# Load the dataset
file_path = "BTC_All_graph_coinmarketcap.csv"
df = pd.read_csv(file_path, delimiter='\;')

# Display basic information and first few rows
df.info(), df.head()


#%%
from datetime import datetime
import matplotlib.pyplot as plt

# Convert time-related columns to datetime format
df.drop(columns=["name"])
df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('"' , '') , format='ISO8601' )
# Sort by timestamp in ascending order
df = df.sort_values(by='timestamp')

# Plot the closing price trend
plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['close'], label='Bitcoin Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('Bitcoin Closing Price Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()






#%%

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Feature engineering
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day

# Selecting features and target variable
features = ['open', 'high', 'low', 'volume', 'marketCap', 'day_of_week', 'month', 'day']
target = 'close'

# Splitting data into train and test sets
X = df[features]
y = df[target]


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

# Select only the 'close' price for prediction
data = df[['close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences (use the past 30 days to predict the next)
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=16, epochs=30, validation_data=(X_test, y_test))

#%% Predict on the test set
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Evaluate the model
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'][train_size + SEQ_LENGTH:], scaler.inverse_transform(y_test), label='Actual Price')
plt.plot(df['timestamp'][train_size + SEQ_LENGTH:], y_pred_rescaled, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Bitcoin Closing Price')
plt.title('Bitcoin Price Prediction with LSTM')
plt.legend()
plt.show()

#%%
# Predict the next day's price
last_30_days = data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
next_day_pred = model.predict(last_30_days)
next_day_price = scaler.inverse_transform(next_day_pred)[0][0]

print(f"Predicted Closing Price for Next Day: ${next_day_price:.2f}")
