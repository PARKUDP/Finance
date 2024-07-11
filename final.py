import keras
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import streamlit as st

class DataLoader:
    def __init__(self, stock_code):
        self.stock_code = stock_code
    
    def _get_time(self):
        dt_now = datetime.now()
        end = dt_now.strftime('%Y-%m-%d')
        start_day = dt_now - timedelta(days=1095)
        start = start_day.strftime('%Y-%m-%d')
        return start, end
    
    def get_data(self):
        start, end = self._get_time()
        df = yf.download(self.stock_code, start=start, end=end, interval='1d')
        data = df.filter(['Close'])
        dataset = data.values
        return data, dataset

class ModelTrainer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0,1))
    
    def normalize_data(self, data):
        return self.scaler.fit_transform(data)
    
    def split_data(self, data, ratio=0.7):
        training_data_len = int(np.ceil(len(data) * ratio))
        train_data = data[0: training_data_len, :]
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        return x_train, y_train, training_data_len
    
    def train_model(self, x_train, y_train):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=16, epochs=20)
        return model
    
    def make_predictions(self, model, data, training_data_len):
        test_data = data[training_data_len - 60:, :]
        x_test = []
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predictions = model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def predict_future(self, model, last_data, days=90):
        future_predictions = []
        last_data = last_data[-60:]
        for _ in range(days):
            input_data = last_data.reshape((1, last_data.shape[0], 1))
            pred = model.predict(input_data)
            future_predictions.append(pred[0, 0])
            last_data = np.append(last_data[1:], pred)
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        return future_predictions

class Plotter:
    @staticmethod
    def plot_data_with_predictions(train, valid, predictions):
        plt.figure(figsize=(16, 6))
        plt.title('LSTM Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.plot(train['Close'], label='Train')
        plt.plot(valid['Close'], label='Real')
        plt.plot(valid.index, predictions, label='Prediction', color='red')
        plt.legend(['Train', 'Real', 'Prediction'], loc='lower right')
        st.pyplot(plt)

    @staticmethod
    def plot_data_with_streamlit(valid):
        st.line_chart(valid)
    
    @staticmethod
    def plot_future_predictions(data, future_predictions, days=90):
        plt.figure(figsize=(16, 6))
        plt.title('Future Predictions')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.plot(data['Close'], label='Historical Data')
        future_dates = pd.date_range(start=data.index[-1], periods=days+1, closed='right')
        plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
        plt.legend(['Historical Data', 'Future Predictions'], loc='lower right')
        st.pyplot(plt)
    
    @staticmethod
    def plot_future_with_streamlit(data, future_predictions, days=90):
        future_dates = pd.date_range(start=data.index[-1], periods=days+1, closed='right')
        future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Close'])
        combined_df = pd.concat([data, future_df])
        st.line_chart(combined_df)

class StreamlitApp:
    def __init__(self):
        self.data_loader = None
        self.model_trainer = ModelTrainer()
        self.plotter = Plotter()
    
    def run(self):
        st.title("株価予測アプリ")
        stock_code = st.text_input('stockコードを入力してください')
        if stock_code:
            self.data_loader = DataLoader(stock_code)
            data, dataset = self.data_loader.get_data()
            st.markdown('### 3年間の株価データ')
            st.line_chart(data)
            
            scaled_data = self.model_trainer.normalize_data(dataset)
            x_train, y_train, training_data_len = self.model_trainer.split_data(scaled_data)
            if st.button('モデルを訓練'):
                with st.spinner('モデルを訓練中...'):
                    model = self.model_trainer.train_model(x_train, y_train)
                    predictions = self.model_trainer.make_predictions(model, scaled_data, training_data_len)
                    future_predictions = self.model_trainer.predict_future(model, scaled_data, days=90)
                    
                train = data[:training_data_len]
                valid = data[training_data_len:].copy()
                valid.loc[:, 'Predictions'] = predictions
                
                # RMSEの計算
                rmse = np.sqrt(mean_squared_error(valid['Close'], predictions))
                st.write(f'予測精度 (RMSE): {rmse:.2f}')
                
                self.plotter.plot_data_with_predictions(train, valid, predictions)
                self.plotter.plot_data_with_streamlit(valid)
                self.plotter.plot_future_with_streamlit(data, future_predictions)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
