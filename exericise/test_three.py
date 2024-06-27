import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import os

# 現在の日時を取得
dt_now = datetime.now()
end = dt_now.strftime('%Y-%m-%d')
start_day = dt_now - timedelta(days=1095)
start = start_day.strftime('%Y-%m-%d')

# 株価データの取得
stock_code = "1321.T"
df = yf.download(stock_code, start=start, end=end, interval='1d')

data = df.filter(["Open", "High", "Low", "Close"])
dataset = data.values

# データの正規化(Close Price)
data_close = df.filter(["Close"])
dataset_close = data_close.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset_close)

training_data_len = int(np.ceil(len(dataset_close) * 0.7))

train_data = scaled_data[0: int(training_data_len), :]

# 訓練データの取得
x_train = []
y_train = []
for i in range(60, len(train_data)): 
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

# 訓練データのreshape
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

# LSTMモデル構築
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練用モデル構築
model.fit(x_train, y_train, batch_size=1, epochs=20)

# 検証用データを取得とデータ変換
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset_close[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 予測値の算出
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# RMSEを利用して予測精度を確認
test_score = np.sqrt(mean_squared_error(y_test, predictions))

train = data_close[: training_data_len]
valid = data_close[training_data_len:]
valid['Predictions'] = predictions

# Streamlitの設定
st.title('株価予測モデル')
st.write(f'テストスコア (RMSE): {test_score:.2f}')

# 株価データのプロット
st.subheader('価格履歴')
plt.figure(figsize=(16,6))
plt.plot(data["Close"], label="Close")
plt.plot(data["Open"], label="Open")
plt.plot(data["High"], label="High")
plt.plot(data["Low"], label="Low")
plt.legend()
st.pyplot(plt)

# LSTMモデルの予測結果プロット
st.subheader('LSTMモデルの予測結果')
plt.figure(figsize=(16,6))
plt.plot(train['Close'], label='Train')
plt.plot(valid['Close'], label='Real')
plt.plot(valid['Predictions'], label='Prediction')
plt.legend(['Train', 'Real', 'Prediction'], loc='lower right')
st.pyplot(plt)
