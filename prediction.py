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


from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

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
 
# define input sequence
raw_seq = fft_100_close
print('==================')
print('==================')
print('LSTM model')
print('==================')
print('==================')



# define input sequence
raw_seq = fft_100_close[len(fft_100_close)-8:len(fft_100_close)]
print(raw_seq)
# choose a number of time steps
n_steps = 7
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array(fft_100_close[len(fft_100_close)-8:len(fft_100_close)-1])
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=0).tolist()[0][0]
print('predicted: ',yhat)
print(type(yhat))
print('expected: 115.16')
