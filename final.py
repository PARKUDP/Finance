import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 現在の日時を取得
dt_now = datetime.datetime.now()
end = dt_now.strftime('%Y-%m-%d')
start_day = dt_now - datetime.timedelta(days=1095)
start = start_day.strftime('%Y-%m-%d')

# 株価データの取得
stock_code = "KO"
ticker = yf.Ticker(stock_code)
hist = ticker.history(start=start, end=end, interval='1d')

# データの整形
df = pd.DataFrame(hist)
df['date'] = df.index
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# 特徴量とターゲットの設定
X = df[['year', 'month', 'day']]
y = df[['Open', 'High', 'Low', 'Close']]

# モデルの学習
model = make_pipeline(StandardScaler(), MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3)))
model.fit(X, y)

# 予測
next_date = dt_now + datetime.timedelta(days=1)
predict_df = pd.DataFrame({
    'year': [next_date.year],
    'month': [next_date.month],
    'day': [next_date.day]
})
predicted_price = model.predict(predict_df)

# 予測結果の出力
print(f"Predicted prices for {next_date.strftime('%Y-%m-%d')}: Open={predicted_price[0][0]}, High={predicted_price[0][1]}, Low={predicted_price[0][2]}, Close={predicted_price[0][3]}")

# 予測結果の可視化
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['Open'], label='Open')
plt.plot(df['date'], df['High'], label='High')
plt.plot(df['date'], df['Low'], label='Low')
plt.plot(df['date'], df['Close'], label='Close')

# 予測データの追加
plt.scatter(next_date, predicted_price[0][0], color='red', label='Predicted Open', zorder=5)
plt.scatter(next_date, predicted_price[0][1], color='red', label='Predicted High', zorder=5)
plt.scatter(next_date, predicted_price[0][2], color='red', label='Predicted Low', zorder=5)
plt.scatter(next_date, predicted_price[0][3], color='red', label='Predicted Close', zorder=5)

plt.title(f"Stock Prices for {stock_code}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()