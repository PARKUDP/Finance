import keras
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import os

# 株価データの取得
stock_code = "1321.T"
df = yf.download(stock_code, start='2016-01-01', end='2024-07-17', interval='1d')

# データセットの分割
data = df.filter(["Open", "High", "Low", "Close"])

# グラフの保存ディレクトリを作成
if not os.path.exists('image'):
    os.makedirs('image')

# dataset plot(Open, High, Low, Close)
plt.figure(figsize=(16,6))
plt.title("Price History")
plt.plot(data["Close"], label="Close")
plt.plot(data["Open"], label="Open")
plt.plot(data["High"], label="High")
plt.plot(data["Low"], label="Low")
plt.legend()
plt.savefig("image/price_history.png")

# データの正規化
data = df.filter(["Close"])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# 2016年1月から2019年1月までのデータを使用してモデルを学習
train_data_16_19 = scaled_data[0:df.index.get_loc('2019-02-01')]
x_train_16_19 = []
y_train_16_19 = []

for i in range(60, len(train_data_16_19)):
    x_train_16_19.append(train_data_16_19[i-60:i, 0])
    y_train_16_19.append(train_data_16_19[i, 0])

x_train_16_19, y_train_16_19 = np.array(x_train_16_19), np.array(y_train_16_19)
x_train_16_19 = np.reshape(x_train_16_19, (x_train_16_19.shape[0], x_train_16_19.shape[1], 1))

# LSTMモデル構築
model_16_19 = Sequential()
model_16_19.add(LSTM(128, return_sequences=True, input_shape=(x_train_16_19.shape[1], 1)))
model_16_19.add(LSTM(64, return_sequences=False))
model_16_19.add(Dense(25))
model_16_19.add(Dense(1))
model_16_19.compile(optimizer='adam', loss='mean_squared_error')

# 訓練用モデル構築
model_16_19.fit(x_train_16_19, y_train_16_19, batch_size=16, epochs=20)

# 2019年2月から2024年7月17日までのデータを使用してモデルを学習
train_data_19_now = scaled_data[df.index.get_loc('2019-02-01'):]
training_data_len_19_now = int(np.ceil(len(train_data_19_now) * 0.7))

train_data_19_now = scaled_data[df.index.get_loc('2019-02-01'):df.index.get_loc('2019-02-01') + training_data_len_19_now]
x_train_19_now = []
y_train_19_now = []

for i in range(60, len(train_data_19_now)):
    x_train_19_now.append(train_data_19_now[i-60:i, 0])
    y_train_19_now.append(train_data_19_now[i, 0])

x_train_19_now, y_train_19_now = np.array(x_train_19_now), np.array(y_train_19_now)
x_train_19_now = np.reshape(x_train_19_now, (x_train_19_now.shape[0], x_train_19_now.shape[1], 1))

# LSTMモデル構築
model_19_now = Sequential()
model_19_now.add(LSTM(128, return_sequences=True, input_shape=(x_train_19_now.shape[1], 1)))
model_19_now.add(LSTM(64, return_sequences=False))
model_19_now.add(Dense(25))
model_19_now.add(Dense(1))
model_19_now.compile(optimizer='adam', loss='mean_squared_error')

# 訓練用モデル構築
model_19_now.fit(x_train_19_now, y_train_19_now, batch_size=16, epochs=20)

# 2019年2月から2024年7月17日までのデータを使用して両方のモデルで予測
test_data_19_now = scaled_data[df.index.get_loc('2019-02-01') + training_data_len_19_now - 60:]
x_test_19_now = []
y_test_19_now = dataset[df.index.get_loc('2019-02-01') + training_data_len_19_now:, :]

for i in range(60, len(test_data_19_now)):
    x_test_19_now.append(test_data_19_now[i-60:i, 0])

x_test_19_now = np.array(x_test_19_now)
x_test_19_now = np.reshape(x_test_19_now, (x_test_19_now.shape[0], x_test_19_now.shape[1], 1))

# 予測値の算出
predictions_19_now_model_16_19 = model_16_19.predict(x_test_19_now)
predictions_19_now_model_16_19 = scaler.inverse_transform(predictions_19_now_model_16_19)

predictions_19_now_model_19_now = model_19_now.predict(x_test_19_now)
predictions_19_now_model_19_now = scaler.inverse_transform(predictions_19_now_model_19_now)

# RMSEを利用して予測精度を確認
test_score_19_now_model_16_19 = np.sqrt(mean_squared_error(y_test_19_now, predictions_19_now_model_16_19))
print('Test Score (2016-2019 model): %.2f RMSE' % (test_score_19_now_model_16_19))

test_score_19_now_model_19_now = np.sqrt(mean_squared_error(y_test_19_now, predictions_19_now_model_19_now))
print('Test Score (2019-2024 model): %.2f RMSE' % (test_score_19_now_model_19_now))

# 最大誤差と最小誤差を計算
errors_19_now_model_16_19 = y_test_19_now - predictions_19_now_model_16_19
max_error_19_now_model_16_19 = np.max(errors_19_now_model_16_19)
min_error_19_now_model_16_19 = np.min(errors_19_now_model_16_19)
print(f'Max Error (2016-2019 model): {max_error_19_now_model_16_19:.2f}')
print(f'Min Error (2016-2019 model): {min_error_19_now_model_16_19:.2f}')

errors_19_now_model_19_now = y_test_19_now - predictions_19_now_model_19_now
max_error_19_now_model_19_now = np.max(errors_19_now_model_19_now)
min_error_19_now_model_19_now = np.min(errors_19_now_model_19_now)
print(f'Max Error (2019-2024 model): {max_error_19_now_model_19_now:.2f}')
print(f'Min Error (2019-2024 model): {min_error_19_now_model_19_now:.2f}')

# 結果をプロット
train_19_now = data[:df.index.get_loc('2019-02-01') + training_data_len_19_now]
valid_19_now = data[df.index.get_loc('2019-02-01') + training_data_len_19_now:].copy()
valid_19_now['Predictions_16_19'] = predictions_19_now_model_16_19
valid_19_now['Predictions_19_now'] = predictions_19_now_model_19_now

plt.figure(figsize=(16,6))
plt.title('LSTM Model Predictions (2019 February to now)')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price of 1321.T', fontsize=18)
plt.plot(train_19_now['Close'], label='Train')
plt.plot(valid_19_now['Close'], label='Real')
plt.plot(valid_19_now['Predictions_16_19'], label='Predictions (2016-2019 model)')
plt.plot(valid_19_now['Predictions_19_now'], label='Predictions (2019-2024 model)')
plt.legend(loc='lower right')
plt.savefig("image/lstm_model_comparison.png")
