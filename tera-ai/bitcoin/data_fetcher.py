# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 02:25:32 2025

@author: HP
"""

import requests
import pandas as pd
import time

def get_binance_klines(symbol="BTCUSDT", interval="1m", limit=1000, start_time=None, end_time=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df["close"] = df["close"].astype(float)
    df.save('btc.csv')
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


get_binance_klines()
