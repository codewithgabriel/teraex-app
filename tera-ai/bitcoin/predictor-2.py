# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 09:52:23 2025

@author: HP
"""

import pandas as pd

# Load the dataset
file_path = "Bitcoin_4_4_2024-4_4_2025_historical_data_coinmarketcap.csv"
df = pd.read_csv(file_path, sep='\;')

# Display basic information and first few rows
df.info(), df.head()


from datetime import datetime
import matplotlib.pyplot as plt

# Convert time-related columns to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('"' , '') , format='ISO8601')

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a Random Forest Regression model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mae


#%%
import pandas as pd
from datetime import timedelta

# Get the last row (latest data)
last_row = df.iloc[-1]

# Estimate next day's features
next_day = last_row['timestamp'] + timedelta(days=2)

next_day_features = {
    'open': last_row['open'],  # Assuming the opening price is the previous close
    'high': last_row['high'],  # Keeping high similar
    'low': last_row['low'],    # Keeping low similar
    'volume': last_row['volume'],  # Keeping volume the same
    'marketCap': last_row['marketCap'],  # Keeping market cap the same
    'day_of_week': next_day.dayofweek,
    'month': next_day.month,
    'day': next_day.day
}


#%% Convert to DataFrame for model prediction
next_day_df = pd.DataFrame([next_day_features])

# Predict closing price
predicted_close = model.predict(next_day_df)[0]

# Show result
print(f"Predicted closing price for {next_day.date()}: ${predicted_close:.2f}")


#%%
import numpy as np

# Get predictions from each tree in the forest
tree_preds = np.array([tree.predict(next_day_df)[0] for tree in model.estimators_])

# Calculate statistics
mean_pred = np.mean(tree_preds)
std_pred = np.std(tree_preds)
ci_lower = mean_pred - 1.96 * std_pred
ci_upper = mean_pred + 1.96 * std_pred

# Display result with confidence interval
print(f"\n📈 Predicted closing price for {next_day.date()}: ${mean_pred:.2f}")
print(f"🧠 95% Confidence Interval: ${ci_lower:.2f} to ${ci_upper:.2f}")
