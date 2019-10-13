import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
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
from keras.models import load_model

# to not display the warnings of tensorflow
import os
import utils

#add a main
#add function to get the original files
#add a function to add the new values
#make the values go through the analysis
#create a dataset containing the values for prediction, low, high plus analysis as input to a new model 

stock_name = 'AI'  #AF   AIR   AI
stock = stock_name+'.csv'

fft_3_high = []
fft_6_high = []
fft_10_high = []
fft_100_high = []

fft_3_low = []
fft_6_low = []
fft_10_low = []
fft_100_low = []

fft_3_close = []
fft_6_close = []
fft_10_close = []
fft_100_close = []

df = utils.format_dataset(stock)

df_fft_close = utils.fourier_transform_close(df,fft_3_close,fft_6_close,fft_10_close,fft_100_close)
df_fft_low = utils.fourier_transform_low(df,fft_3_low,fft_6_low,fft_10_low, fft_100_low)
df_fft_high = utils.fourier_transform_high(df,fft_3_high,fft_6_high,fft_10_high, fft_100_high)

fft_3_close = list(itertools.chain.from_iterable(fft_3_close))
fft_6_close = list(itertools.chain.from_iterable(fft_6_close))
fft_10_close = list(itertools.chain.from_iterable(fft_10_close))
fft_100_close = list(itertools.chain.from_iterable(fft_100_close))

fft_3_high = list(itertools.chain.from_iterable(fft_3_high))
fft_6_high = list(itertools.chain.from_iterable(fft_6_high))
fft_10_high = list(itertools.chain.from_iterable(fft_10_high))
fft_100_high = list(itertools.chain.from_iterable(fft_100_high))

fft_3_low = list(itertools.chain.from_iterable(fft_3_low))
fft_6_low = list(itertools.chain.from_iterable(fft_6_low))
fft_10_low = list(itertools.chain.from_iterable(fft_10_low))
fft_100_low = list(itertools.chain.from_iterable(fft_100_low))



df_dict = {'index': list(range(0,len(df))),'name':df.name.values.tolist(), 'date':df.date.values.tolist(), 'volume':df.volume.values.tolist(), 'fft_3_close':fft_3_close,'fft_6_close':fft_6_close,'fft_10_close':fft_10_close, 'fft_100_close':fft_100_close, 'fft_3_high':fft_3_high,'fft_6_high':fft_6_high,'fft_10_high':fft_10_high,'fft_100_high':fft_100_high,'fft_3_low':fft_3_low , 'fft_6_low':fft_6_low,'fft_10_low':fft_10_low, 'fft_100_low':fft_100_low}
df = pd.DataFrame(df_dict) 
df = utils.get_technical_indicators(df)
df2 = utils.get_technical_indicators(df)
df = df[['date', 'fft_3_close','fft_6_close','fft_10_close','fft_100_close', 'fft_3_low','fft_6_low','fft_10_low', 'fft_100_low', 'fft_3_high','fft_6_high','fft_10_high','fft_100_high', 'volume', 'var_mma', 'var_macd','var_bollinger', 'ema','upper_band', 'lower_band']]
pct_close_mvt = utils.get_movement(df,fft_100_close)
fft = df.fft_100_close.values.tolist()


momentum = []
momentum = utils.generate_momentum(fft,momentum)

i = 0
fft = df.fft_100_close.values.tolist()
var_bollinger = df.var_bollinger.values.tolist()

variation = []
i = 1
while i < len(var_bollinger):
    variation.append(var_bollinger[i]-var_bollinger[i-1])
    i += 1

upper = df.upper_band.values.tolist()
lower = df.lower_band.values.tolist()
stock_bollinger_upper = []
stock_bollinger_lower = []
value_bollinger = []
i = 0
while i < len(fft):
    stock_bollinger_upper.append(fft[i]-upper[i])
    stock_bollinger_lower.append(fft[i]-lower[i])
    i += 1
    
i = 0
while i < len(fft):
    if str(stock_bollinger_upper[i]) == 'nan':
        value_bollinger.append(100)
    elif stock_bollinger_upper[i] > 0:
        value_bollinger.append(1)
    elif stock_bollinger_lower[i] < 0:
        value_bollinger.append(-1)
    elif stock_bollinger_upper[i] < 0 and  stock_bollinger_lower[i] > 0:
        value_bollinger.append(0)
    i += 1

value_macd = []
var_macd_list = df.var_macd.values.tolist()
i = 0
while i < len(var_macd_list):
    if var_macd_list[i] > 0:
        value_macd.append(1)
    elif var_macd_list[i] < 1:
        value_macd.append(-1)
    else:
        value_macd.append(0)
    i += 1

value_mma = []
var_mma_list = df.var_mma.values.tolist()
i = 0
while i < len(var_mma_list):
    if var_mma_list[i] > 0:
        value_mma.append(1)
    elif var_mma_list[i] < 1:
        value_mma.append(-1)
    else:
        value_mma.append(0)
    i += 1

df_dict = {'date':df.date.values.tolist(),'pct_close_mvt':pct_close_mvt,'fft_3_close':df.fft_3_close.values.tolist(),'fft_6_close':df.fft_6_close.values.tolist(),'fft_10_close':df.fft_10_close.values.tolist(),'fft_3_low':df.fft_3_low.values.tolist(),'fft_6_low':df.fft_6_low.values.tolist(),'fft_10_low':df.fft_10_low.values.tolist(),'fft_3_high':df.fft_3_high.values.tolist(),'fft_6_high': df.fft_6_high.values.tolist(),'fft_10_high':df.fft_10_high.values.tolist(),'fft_100_close': df.fft_100_close.values.tolist(),'fft_100_low': df.fft_100_low.values.tolist(),'fft_100_high': df.fft_100_high.values.tolist(),'value_bollinger':value_bollinger,'value_macd':value_macd, 'value_mma':value_mma,'var_bollinger':df.var_bollinger.values.tolist(), 'momentum':momentum}
df = pd.DataFrame(df_dict) 

momentum = df.momentum.values.tolist()

label = []
i = 0
while i < len(momentum)-1:
    if momentum[i] == 0 and momentum[i+1] == 0:
        label.append(0)
    elif momentum[i] == 1 and momentum[i+1] == 1:
        label.append(0)
    elif momentum[i] == 1 and momentum[i+1] == 0:
        label.append(1)
    elif momentum[i] == 0 and momentum[i+1] == 1:
        label.append(2)
    i += 1
if momentum[241] == 0 and momentum[242] == 0:
    label.append(0)
elif momentum[241] == 1 and momentum[242] == 1:
    label.append(0)
elif momentum[241] == 0 and momentum[242] == 1:
    label.append(1)
elif momentum[241] == 1 and momentum[242] == 0:
    label.append(2)


np.random.seed(7)
look_back = 14
epochs = 10
batch_size = 64

stock_prices = df['fft_100_close'].values.astype('float32')
stock_prices = stock_prices.reshape(len(stock_prices), 1)

#create a generic function for training the models
# split data into training set and test set
test_predict_close = utils.prediction_models(df, 'fft_100_close',look_back,epochs, batch_size,'close.h5')
model = load_model('close.h5')

#stock_prices_close = utils.format_data(df,'fft_100_high')
test_predict_low = utils.prediction_models(df, 'fft_100_low',look_back,epochs, batch_size,'low.h5')
print(test_predict_low)

#stock_prices_close = utils.format_data(df,'fft_100_low')
test_predict_high = utils.prediction_models(df, 'fft_100_high',look_back,epochs, batch_size,'high.h5')
print(test_predict_high)




#plt.figure(figsize=(20,10))
#plt.plot(df2.date.values.tolist(),df2.fft_100_close.values.tolist(), label='fft100_close')
#plt.plot(df2.date.values.tolist(),df2.amd20.values.tolist(), label='ema20')
#plt.plot(df2.date.values.tolist(),df2.amd50.values.tolist(), label='ema50')
#plt.plot(df2.date.values.tolist(),df2.upper_band.values.tolist(), label='upper')
#plt.plot(df2.date.values.tolist(),df2.lower_band.values.tolist(), label='lower')
#plt.legend()
#plt.show()

#plt.figure(figsize=(20,10))
#plt.plot(df2.date.values.tolist(),df2.MACD.values.tolist(), label='MACD')
#plt.plot(df2.date.values.tolist(),df2.signal.values.tolist(), label='signal')
#plt.legend()
#plt.show()

#plt.figure(figsize=(20,10))
#plt.plot(df2.date.values.tolist(),df2.fft_100_low.values.tolist(), label='fft100_low')
#plt.plot(df2.date.values.tolist(),df2.fft_10_low.values.tolist(), label='fft_10_low')
#plt.plot(df2.date.values.tolist(),df2.fft_6_low.values.tolist(), label='fft_6_low')
#plt.plot(df2.date.values.tolist(),df2.fft_3_low.values.tolist(), label='fft_3_low')
#plt.legend()
#plt.show()

#plt.figure(figsize=(20,10))
#plt.plot(df2.date.values.tolist(),df2.fft_100_high.values.tolist(), label='fft100_high')
#plt.plot(df2.date.values.tolist(),df2.fft_10_high.values.tolist(), label='fft_10_high')
#plt.plot(df2.date.values.tolist(),df2.fft_6_high.values.tolist(), label='fft_6_high')
#plt.plot(df2.date.values.tolist(),df2.fft_3_high.values.tolist(), label='fft_3_high')
#plt.legend()
#plt.show()

#plt.figure(figsize=(20,10))
#plt.plot(df2.date.values.tolist(),df2.fft_100_close.values.tolist(), label='fft100_close')
#plt.plot(df2.date.values.tolist(),df2.fft_100_high.values.tolist(), label='fft_100_high')
#plt.plot(df2.date.values.tolist(),df2.fft_100_low.values.tolist(), label='fft_100_low')
#plt.legend()
#plt.show()





