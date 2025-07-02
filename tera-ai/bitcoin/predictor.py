# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 07:06:48 2025

@author: HP
"""

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#%% Load and preprocess the dataset
dataset = pd.read_csv("Bitcoin_4_4_2024-4_4_2025_historical_data_coinmarketcap.csv", sep=';')
dataset = dataset.drop(columns=["name"])
dataset["timeOpen"] = pd.to_datetime(dataset["timeOpen"].str.replace('"', ''))
dataset["timeClose"] = pd.to_datetime(dataset["timeClose"].str.replace('"', ''))
dataset["timeHigh"] = pd.to_datetime(dataset["timeHigh"].str.replace('"', ''))
dataset["timeLow"] = pd.to_datetime(dataset["timeLow"].str.replace('"', ''))

# Convert datetime columns to Unix timestamps (seconds since epoch)
dataset["timeOpen_unix"] = dataset["timeOpen"].view('int64') // 10**9  # Convert to seconds
dataset["timeClose_unix"] = dataset["timeClose"].view('int64') // 10**9
dataset["timeHigh_unix"] = dataset["timeHigh"].view('int64') // 10**9
dataset["timeLow_unix"] = dataset["timeLow"].view('int64') // 10**9

dataset["timestamp"] = pd.to_datetime(dataset["timestamp"].str.replace('"', ''))

#%% Extract meaningful numerical features from datetime
dataset["hour"] = dataset["timestamp"].dt.hour
dataset["day"] = dataset["timestamp"].dt.day
dataset["month"] = dataset["timestamp"].dt.month
dataset["year"] = dataset["timestamp"].dt.year

#%% Define features (X) and target variable (y) for prediction
X = dataset[["hour", "day", "month", "year", "timeOpen_unix", "timeClose_unix", "timeHigh_unix", "timeLow_unix"]].values
y = dataset["close"].values  # Assuming "close" is the column you're predicting.

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Train the linear regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#%% Make predictions
pred = linreg.predict(X_test)

# Print predictions and model score
print("Predictions:", pred)
print("Model Score:", linreg.score(X_test, y_test))


#%% using decision tree regression model 

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train the Decision Tree Regressor
dtreg = DecisionTreeRegressor() 
dtreg.fit(X_train, y_train)

# Make predictions
dt_pred = dtreg.predict(X_test)

# Evaluate the model with metrics
mae = mean_absolute_error(y_test, dt_pred)
mse = mean_squared_error(y_test, dt_pred)
r2 = r2_score(y_test, dt_pred)

# Print the metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)


#%% Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train the Random Forest Regressor
rfreg = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators (number of trees)
rfreg.fit(X_train, y_train)

#%% Make predictions
rf_pred = rfreg.predict(X_test)

# Evaluate the model with metrics
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Print the metrics
print("Random Forest Metrics:")
print("Mean Absolute Error (MAE):", rf_mae)
print("Mean Squared Error (MSE):", rf_mse)
print("R-squared (R²):", rf_r2)

#%%

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate future timestamps for the next 2 days (48 hours)
future_timestamps = [datetime.now() + timedelta(hours=i) for i in range(48)]

# Create a DataFrame for future predictions
future_data = pd.DataFrame({
    "timestamp": future_timestamps
})

# Extract numerical features from future timestamps
future_data["hour"] = future_data["timestamp"].dt.hour
future_data["day"] = future_data["timestamp"].dt.day
future_data["month"] = future_data["timestamp"].dt.month
future_data["year"] = future_data["timestamp"].dt.year

# Calculate mean values for the required features
timeOpen_mean = dataset["timeOpen_unix"].mean()
timeClose_mean = dataset["timeClose_unix"].mean()
timeHigh_mean = dataset["timeHigh_unix"].mean()
timeLow_mean = dataset["timeLow_unix"].mean()

# Add placeholder features to future_data
future_data["timeOpen_unix"] = timeOpen_mean
future_data["timeClose_unix"] = timeClose_mean
future_data["timeHigh_unix"] = timeHigh_mean
future_data["timeLow_unix"] = timeLow_mean

# Select all 8 features for predictions
future_features = future_data[["hour", "day", "month", "year", "timeOpen_unix", "timeClose_unix", "timeHigh_unix", "timeLow_unix"]].values

# Use the trained model for predictions
rf_future_pred = rfreg.predict(future_features)

# Add predictions to the DataFrame
future_data["RandomForest_Predictions"] = rf_future_pred

# Print the future predictions
print(future_data)
