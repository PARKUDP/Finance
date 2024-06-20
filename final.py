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
import os
import streamlit as st 


class Finance:
    def __init__(self):
        pass
    
    def _get_time(self):
        self.dt_now = datetime.now()
        self.end = self.dt_now.strftime('%Y-%m-%d')
        self.start_day = self.dt_now - timedelta(days=1095)
        self.start = self.start_day.strftime('%Y-%m-%d')
        
        return self.start, self.end

    def get_data(self, stock_code):
        self.start_day, self.end_day = self._get_time()
        self.df = yf.download(stock_code, start=self.start_day, end=self.end_day, interval='1d')
        self.data = self.df.filter(['Close'])
        self.dataset = self.data.values
        
        return self.dataset

    def plot_data(self, x, y, title, xlabel, ylabel, save_path):
        plt.figure(figsize=(16,6))
        plt.title(title)
        plt.plot(x, y)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.savefig(save_path)
        
    def normalize_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = self.scaler.fit_transform(data)
        
        return self.scaled_data
    
    def split_data(self, data, ratio):
        self.training_data_len = int(np.ceil(len(data) * ratio))
        self.train_data = data[0: int(self.training_data_len), :]
        self.x_train = []
        self.y_train = []
        for i in range(60, len(self.train_data)): 
            self.x_train.append(self.train_data[i-60:i,0])
            self.y_train.append(self.train_data[i,0])

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train,(self.x_train.shape[0], self.x_train.shape[1], 1))
        
        self.train = self.data[:self.training_data_len]
        self.vaild = self.data[self.training_data_len:]

        return self.x_train, self.y_train, self.train, self.vaild
    
    def train_model(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.x_train, self.y_train, batch_size=1, epochs=1)
        
        return self.model


finance = Finance()
stock_code = st.text_input('Enter stock code')
data = finance.get_data(stock_code)
data = finance.normalize_data(data)
x_train, y_train = finance.split_data(data, 0.7)
model = finance.train_model(x_train, y_train)
