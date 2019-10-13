import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

# to not display the warnings of tensorflow
import os
import utils


def format_number(s):
    return float(s.replace(',', '.'))

def date_corrector(s):
    return str(s[0:6]+'20'+s[6:8])

def format_dataset(stock):
    df = pd.read_csv(stock,delimiter=';')
    df = df.rename(index=str, columns={df.columns[0]:"name",df.columns[1]:"date",df.columns[2]:"open",df.columns[3]:"high",df.columns[4]:"low",df.columns[5]:"close",df.columns[6]:"volume"})
    df.open = list(map(format_number, df.open))
    df.high = list(map(format_number, df.high))
    df.low = list(map(format_number, df.low))
    df.close = list(map(format_number, df.close))
    df.date = list(map(date_corrector,df.date))
    return df

def get_movement(df,fft_100_close):
    i = 1
    close_value = df.fft_100_close.values.tolist()
    pct_close_mvt = [0]
    while i < len(close_value):
        pct_close_mvt.append((fft_100_close[i]-fft_100_close[i-1])/fft_100_close[i-1]*100.0)
        i += 1
    return pct_close_mvt
    
def get_technical_indicators(df):
    # Create 7 and 21 days Moving Average
    df['ma20'] = df['fft_100_close'].rolling(window=20).mean()
    df['ma50'] = df['fft_100_close'].rolling(window=50).mean()
    df['ma150'] = df['fft_100_close'].rolling(window=150).mean()
    df['amd20'] = df['fft_100_close'].ewm(span=20,adjust=False).mean()
    df['amd50'] = df['fft_100_close'].ewm(span=50,adjust=False).mean()
    df['var_mma'] = (df['ma50']-df['ma20'])
    df['var_amd'] = (df['amd50']-df['amd20'])
    # Create MACD
    df['26ema'] = pd.ewma(df['fft_100_close'], span=26)
    df['12ema'] = pd.ewma(df['fft_100_close'], span=12)
    df['MACD'] = (df['12ema']-df['26ema'])
    df['signal'] = pd.ewma(df['MACD'], span=9)
    df['var_macd'] = (df['MACD']-df['signal'])
    # Create Bollinger Bands
    df['20sd'] = pd.stats.moments.rolling_std(df['fft_100_close'],20)
    df['ma21'] = df['fft_100_close'].rolling(window=21).mean()
    df['upper_band'] = df['ma21'] + (df['20sd']*2)
    df['lower_band'] = df['ma21'] - (df['20sd']*2)
    df['var_bollinger'] = df['upper_band']- df['lower_band']
    # Create Exponential moving average
    df['ema'] = df['MACD'].ewm(com=0.5).mean()
    return df

def fourier_transform_close(df,fft_3_close,fft_6_close,fft_10_close,fft_100_close):
    data_FT = df[['date', 'close']]
    close_fft = np.fft.fft(np.asarray(data_FT['close'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3,6,10,100]:
        fft_list_m10=np.copy(fft_list); fft_list_m10[num_:-num_]=0
        if num_ == 3:
            fft_3_close.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        if num_ == 6:
            fft_6_close.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        if num_ == 10:
            fft_10_close.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        if num_ == 100:
            fft_100_close.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
    return(fft_df)

def fourier_transform_low(df,fft_3_low,fft_6_low,fft_10_low,fft_100_low):
    data_FT = df[['date', 'low']]
    low_fft = np.fft.fft(np.asarray(data_FT['low'].tolist()))
    fft_df = pd.DataFrame({'fft':low_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3,6,10,100]:
        fft_list_m10=np.copy(fft_list); fft_list_m10[num_:-num_]=0
        if num_ == 3:
            fft_3_low.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        if num_ == 6:
            fft_6_low.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        if num_ == 10:
            fft_10_low.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        fft_list_m10=np.copy(fft_list); fft_list_m10[num_:-num_]=0
        if num_ == 100:
            fft_100_low.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
    return(fft_df)

def fourier_transform_high(df,fft_3_high,fft_6_high,fft_10_high,fft_100_high):
    data_FT = df[['date', 'high']]
    high_fft = np.fft.fft(np.asarray(data_FT['high'].tolist()))
    fft_df = pd.DataFrame({'fft':high_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3,6,10,100]:
        fft_list_m10=np.copy(fft_list); fft_list_m10[num_:-num_]=0
        if num_ == 3:
            fft_3_high.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        if num_ == 6:
            fft_6_high.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        if num_ == 10:
            fft_10_high.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
        fft_list_m10=np.copy(fft_list); fft_list_m10[num_:-num_]=0
        if num_ == 100:
            fft_100_high.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
    return(fft_df)

def generate_momentum(fft,momentum):
    i = 0
    while i < len(fft):
        momentum.append(fft[i]-fft[i-1])
        i += 1
    i = 0
    while i < len(momentum):
        if momentum[i] > 0:
            momentum[i] = 1
        else:
            momentum[i] = 0
        i += 1
    return(momentum)

#Create a function to process the data into 7 day look back slices
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def format_data(df, fft_to_format):
    stock_prices = df[fft_to_format].values.astype('float32')
    stock_prices = stock_prices.reshape(len(stock_prices), 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_prices = scaler.fit_transform(stock_prices)
    return(stock_prices)

def prediction_models(df,stock_prices,look_back, epochs, batch_size, model_name):
    stock_prices = df[stock_prices].values.astype('float32')
    stock_prices = stock_prices.reshape(len(stock_prices), 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_prices = scaler.fit_transform(stock_prices)
    train_size = int(len(stock_prices) * 0.90)
    test_size = len(stock_prices) - train_size
    train, test = stock_prices[0:train_size,:], stock_prices[train_size:len(stock_prices),:]
    print('Split data into training set and test set... Number of training samples/ test samples:', len(train), len(test))
    # convert Apple's stock price data into time series dataset
    trainX, trainY = utils.create_dataset(train, look_back)
    testX, testY = utils.create_dataset(test, look_back)
    # reshape input of the LSTM to be format [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
    # shift predictions of training data for plotting
    trainPredictPlot = np.empty_like(stock_prices)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift predictions of test data for plotting
    testPredictPlot = np.empty_like(stock_prices)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(stock_prices)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(stock_prices))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    model.save(model_name)
    return(testPredict)
