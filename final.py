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
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)
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
            st.line_chart(data)
            
            scaled_data = self.model_trainer.normalize_data(dataset)
            x_train, y_train, training_data_len = self.model_trainer.split_data(scaled_data)
            if st.button('モデルを訓練'):
                model = self.model_trainer.train_model(x_train, y_train)
                predictions = self.model_trainer.make_predictions(model, scaled_data, training_data_len)
                train = data[:training_data_len]
                valid = data[training_data_len:]
                valid['Predictions'] = predictions
                self.plotter.plot_data_with_predictions(train, valid, predictions)
                self.plotter.plot_data_with_streamlit(valid)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
