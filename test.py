import yfinance as yf
import pandas as pd
import numpy as np

# 株価データの取得
stock_code = "KO"
ticker = yf.Ticker(stock_code)
hist = ticker.history(period='3y')
print(hist.columns)