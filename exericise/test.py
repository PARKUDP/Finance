import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2, random_state=0)

# テストデータの日付情報を保持
test_dates = df['date'][test_data.index]

# インデックスをリセット
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
train_target = train_target.reset_index(drop=True)
test_target = test_target.reset_index(drop=True)
test_dates = test_dates.reset_index(drop=True)

model.fit(train_data, train_target)

# 予測
predicted_price = model.predict(test_data)

# 予測の評価
mse = mean_squared_error(test_target, predicted_price)
r2 = r2_score(test_target, predicted_price)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 結果のプロット
plt.figure(figsize=(14, 10))

for i, col in enumerate(y.columns):
    plt.subplot(2, 2, i+1)
    plt.plot(test_dates, test_target[col], label='Actual', color='blue')
    plt.plot(test_dates, predicted_price[:, i], label='Predicted', color='red')
    plt.title(col)
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
