import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
from mpl_finance import candlestick_ohlc

# to not display the warnings of tensorflow
import os

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
    

def get_technical_indicators(df):																	# Create 7 and 21 days Moving Average
	df['ma20'] = df['fft_20_close'].rolling(window=20).mean()
	df['ma50'] = df['fft_20_close'].rolling(window=50).mean()
	#df['ma150'] = df[fft_100].rolling(window=150).mean()
	#df['amd20'] = df[fft_100].ewm(span=20,adjust=False).mean()
	#df['amd50'] = df[fft_100].ewm(span=50,adjust=False).mean()
	#df['var_mma'] = (df['ma50']-df['ma20'])
	#df['var_amd'] = (df['amd50']-df['amd20'])
	# Create MACD
	df['26ema'] = pd.ewma(df['fft_20_close'], span=26)
	df['12ema'] = pd.ewma(df['fft_20_close'], span=12)
	df['MACD'] = (df['12ema']-df['26ema'])
	df['signal'] = pd.ewma(df['MACD'], span=9)
	df['var_macd'] = (df['MACD']-df['signal'])
	# Create Bollinger Bands
	df['20sd'] = pd.stats.moments.rolling_std(df['fft_20_close'],20)
	df['ma21'] = df['fft_20_close'].rolling(window=21).mean()
	df['upper_band'] = df['ma21'] + (df['20sd']*2)
	df['lower_band'] = df['ma21'] - (df['20sd']*2)
	df['var_bollinger'] = df['upper_band']- df['lower_band']
	df['%K'] = STOK(df['fft_20_close'], df['fft_20_low'], df['fft_20_high'], 14)
	df['%D'] = STOD(df['fft_20_close'], df['fft_20_low'], df['fft_20_high'], 14)
	df['var_ema'] = df['26ema'] - df['12ema']
	# Create Exponential moving average
	return df								


def bollinger_indicator(upper_band,lower_band,fft_20_close):
	bollinger_indicator = []
	i = 0
	while i < len(fft_20_close):
		if fft_20_close[i] >= upper_band[i]:
			bollinger_indicator.append(1)
		elif fft_20_close[i] <= lower_band[i]:
			bollinger_indicator.append(-1)
		else:
			bollinger_indicator.append(0)
		i += 1
	return(bollinger_indicator)


def computeRSI(data, time_window):
	diff = data.diff(1).dropna()       
	up_chg = 0 * diff
	down_chg = 0 * diff
	up_chg[diff > 0] = diff[ diff > 0 ]
	down_chg[diff < 0] = diff[ diff < 0 ]
	up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
	down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
	rs = abs(up_chg_avg/down_chg_avg)
	rsi = 100 - 100/(1+rs)
	return rsi

def STOK(close, low, high, n): 
    STOK = ((close - pd.rolling_min(low, n)) / (pd.rolling_max(high, n) -    pd.rolling_min(low, n))) * 100
    return STOK

def STOD(close, low, high, n):
    STOK = ((close - pd.rolling_min(low, n)) / (pd.rolling_max(high, n) - pd.rolling_min(low, n))) * 100
    STOD = pd.rolling_mean(STOK, 3)
    return STOD


def fourier_transform(df,column,fft_20,fft_100):
	data_FT = df[['date', column]]
	fft = np.fft.fft(np.asarray(data_FT[column].tolist()))
	fft_df = pd.DataFrame({'fft':fft})
	fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
	fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
	plt.figure(figsize=(14, 7), dpi=100)
	fft_list = np.asarray(fft_df['fft'].tolist())
	for num_ in [20,100]:
		if num_ == 20:
			fft_list_m10=np.copy(fft_list); fft_list_m10[num_:-num_]=0
			fft_20.append(list(np.fft.ifft(fft_list_m10).real.tolist()))	
		else:	
			fft_list_m10=np.copy(fft_list); fft_list_m10[num_:-num_]=0
			fft_100.append(list(np.fft.ifft(fft_list_m10).real.tolist()))
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
#def processData(data,lb):
#    X,Y = [],[]
#    for i in range(len(data)-lb-1):
#        X.append(data[i:(i+lb),0])
#        Y.append(data[(i+lb),0])
#    return np.array(X),np.array(Y)


#def create_dataset(dataset, look_back):
#    dataX, dataY = [], []
#    for i in range(len(dataset)-look_back-1):
#        a = dataset[i:(i+look_back), 0]
#        dataX.append(a)
#        dataY.append(dataset[i + look_back, 0])
#    return np.array(dataX), np.array(dataY)


#def format_data(df, fft_to_format):
#    stock_prices = df[fft_to_format].values.astype('float32')
#    stock_prices = stock_prices.reshape(len(stock_prices), 1)
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    stock_prices = scaler.fit_transform(stock_prices)
#    return(stock_prices)


#def trend_estimation(fft20_low,fft20_high,variation_low,variation_high,variation):
#    i = 1
#    while i < len(fft20_low):
#        if fft20_low[i]-fft20_low[i-1] > 0:
#            variation_low.append(1)
#        else:
#            variation_low.append(0)
#        i += 1
#    i = 1
#    while i < len(fft20_high):
#        if fft20_high[i]-fft20_high[i-1] > 0:
#            variation_high.append(1)
#        else:
#            variation_high.append(0)
#        i += 1
#    i = 0
#    while i < len(variation_high):
#        if variation_low[i] == 0 and variation_high[i] == 0:
#            variation.append(-1)
#        elif variation_low[i] == 1 and variation_high[i] == 1:
#            variation.append(1)
#        else:
#            variation.append(0)
#        i += 1
#    return(variation)


def stoch_indicator(k_list,d_list):
	stoch_indicator = [0]
	i = 1
	while i < len(k_list):
		if k_list[i] >= 80 and d_list[i] >= 80:
			if k_list[i-1] >= d_list[i-1] and k_list[i] <= d_list[i]:
				stoch_indicator.append(1)
			else:
				stoch_indicator.append(0)
		elif k_list[i] <= 20 and d_list[i] <= 20:
			if k_list[i-1] <= d_list[i-1] and k_list[i] >= d_list[i]:
				stoch_indicator.append(-1)
			else:
				stoch_indicator.append(0)
		else:
			stoch_indicator.append(0)
		i += 1
	return(stoch_indicator)


def ema_indicator(ema_12,ema_26,index,fft_20_close):
	ema_indicator = [0]
	i = 1
	while i < len(fft_20_close):
		if ema_12[i-1] < ema_26[i-1] and ema_26[i] < ema_12[i]:
			ema_indicator.append(-1)
		elif ema_26[i-1] < ema_12[i-1] and ema_12[i] < ema_26[i]:
			ema_indicator.append(1)
		else:
			ema_indicator.append(0)
			pass
		i += 1
	return(ema_indicator)


def ema_trade_indicator(ema_12,ema_26,index,fft_20_close,x,y):
	i = 1
	while i < len(fft_20_close):
		if ema_12[i-1] < ema_26[i-1] and ema_26[i] < ema_12[i]:
			x.append(index[i])
			y.append(fft_20_close[i])
		elif ema_26[i-1] < ema_12[i-1] and ema_12[i] < ema_26[i]:
			x.append(index[i])
			y.append(fft_20_close[i])
		else:
			pass
		i += 1

def bollinger_trade_indicator(bollinger,index,fft_20_close,x,y):
	i = 0
	while i < len(fft_20_close):
		if bollinger[i] == 1:
			x.append(index[i])
			y.append(fft_20_close[i])
		elif bollinger[i] == -1:
			x.append(index[i])
			y.append(fft_20_close[i])
		else:
			pass
		i += 1


def rsi_stoch_trade_indicator(index,rsi,stoch,fft_value,x,y):
	i = 0
	while i < len(rsi):
		if rsi[i] == 1:
			x.append(index[i])
			y.append(fft_value[i])
		elif stoch[i] == 1:
			x.append(index[i])
			y.append(fft_value[i])
		elif rsi[i] == -1:
			x.append(index[i])
			y.append(fft_value[i])
		elif stoch[i] == -1:
			x.append(index[i])
			y.append(fft_value[i])
		else:
			pass
		i += 1


#def compute_stochastic(k_list,d_list,k_value,d_value):
#    i = 0
#    while i < len(k_list):
#        if k_list[i] > 100:
#            k_indicator.append(1)
#            k_value.append(k_list[i])
#        elif k_list[i] < 20:
#            k_indicator.append(-1)
#            k_value.append(k_list[i])
#        else:
#            k_indicator.append(0)
#            k_value.append(0)
#        if d_list[i] > 100:
#            d_indicator.append(1)
#            d_value.append(d_list[i])
#        elif d_list[i] < 20:
#            d_indicator.append(-1)
#            d_value.append(d_list[i])
#        else:
#            d_indicator.append(0)
#            d_value.append(0)
#        i += 1


#def rsi_indicator(rsi_list,rsi):
#    i = 0
#    while i < len(rsi_list):
#        if rsi_list[i] > 100:
#            rsi.append(1)
#        elif rsi_list[i] < 20:
#            rsi.append(-1)
#        else:
#            rsi.append(0)
#        i += 1
#    return(rsi)
def rsi_indicator(rsi_list):
	rsi_indicator = []
	i = 0
	while i < len(rsi_list):
		if rsi_list[i] >= 80:
			rsi_indicator.append(1)
		elif rsi_list[i] <= 20:
			rsi_indicator.append(-1)
		else:
			rsi_indicator.append(0)
		i += 1
	return(rsi_indicator)



def compute_macd(var_macd_list,macd):
	i = 0
	while i < len(var_macd_list):
		if var_macd_list[i] > 0:
			macd.append(1)
		elif var_macd_list[i] < 1:
			macd.append(-1)
		else:
			macd.append(0)
		i += 1
	return(macd)    


def compute_mma(var_mma_list,mma):
	i = 0
	while i < len(var_mma_list):
		if var_mma_list[i] > 0:
			mma.append(1)
		elif var_mma_list[i] < 1:
			mma.append(-1)
		else:
			mma.append(0)
		i += 1
	return(mma)



def generate_graph(df,u,v,w,x,y,z):
	fig = plt.figure(figsize = (20, 5))
	plt.title('stochastich chart')
	df.plot(y=['fft_100_close'])
	df.plot(y=['%K', '%D'], figsize = (20, 5))
	plt.axhline(0, linestyle='--', alpha=0.1)
	plt.axhline(20, linestyle='--', alpha=0.5)
	plt.axhline(30, linestyle='--')
	plt.axhline(70, linestyle='--')
	plt.axhline(80, linestyle='--', alpha=0.5)
	plt.axhline(100, linestyle='--', alpha=0.1)
	plt.savefig('stoch.png')

	fig = plt.figure(figsize = (20, 5))
	plt.figure(figsize=(20,5))
	plt.title('RSI chart')
	plt.plot(df['index'], df['RSI'])
	plt.axhline(0, linestyle='--', alpha=0.1)
	plt.axhline(20, linestyle='--', alpha=0.5)
	plt.axhline(30, linestyle='--')
	plt.axhline(70, linestyle='--')
	plt.axhline(80, linestyle='--', alpha=0.5)
	plt.axhline(100, linestyle='--', alpha=0.1)
	plt.savefig('rsi.png')
			
	fig = plt.figure(figsize = (20, 5))
	plt.figure(figsize=(20,5))
	plt.scatter(w,x, label='event', color='r', s=25, marker="o")
	plt.scatter(y,z, label='event', color='g', s=25, marker="o")
	plt.scatter(u,v, label='event', color='b', s=25, marker="o")
	#plt.plot(df['index'],df['fft_100_close'],label='fft_100_close')
	plt.plot(df['index'],df['fft_20_close'],label='fft_20_close')
	#plt.plot(df['index'],df['ma20'],label='moving average 20')
	#plt.plot(df['index'],df['ma50'],label='moving average 50')
	plt.plot(df['index'],df['26ema'],label='moving average 26')
	plt.plot(df['index'],df['12ema'],label='moving average 12')
	plt.plot(df['index'],df['lower_band'],label='lower')
	plt.plot(df['index'],df['upper_band'],label='upper')
	plt.legend()
	plt.savefig('stock.png', dpi=fig.dpi)

	fig = plt.figure(figsize = (20, 5))
	plt.figure(figsize=(20,5))
	plt.scatter(y,z, label='event', color='k', s=25, marker="o")
	plt.plot(df['index'],df['fft_100_close'],label='fft_100_close')
	plt.plot(df['index'],df['fft_20_close'],label='fft_20_close')
	plt.plot(df['index'],df['ma20'],label='moving average 20')
	plt.plot(df['index'],df['ma50'],label='moving average 50')
	plt.legend()
	plt.savefig('stock20.png', dpi=fig.dpi)

	ohlc = df[['index', 'fft_20_open', 'fft_20_high', 'fft_20_low', 'fft_20_close']]
	fig = plt.figure(figsize=(20,5))
	fig, ax = plt.subplots()
	candlestick_ohlc(ax, ohlc.values, width=0.4,colorup='g', colordown='r');
	ax.set_xlabel('Date')
	ax.set_ylabel('Price')
	plt.savefig("ohlc.png")

	fig = plt.figure(figsize=(20,5))
	fig, ax = plt.subplots()
	candlestick_ohlc(ax, ohlc.values[-5:len(df.index.values.tolist())], width=0.4,colorup='g', colordown='r');
	ax.set_xlabel('Date')
	ax.set_ylabel('Price')
	plt.savefig("ohlc0.png")

	fig = plt.figure(figsize=(20,5))
	fig, ax = plt.subplots()
	candlestick_ohlc(ax, ohlc.values[-1:len(df.index.values.tolist())], width=0.4,colorup='g', colordown='r');
	ax.set_xlabel('Date')
	ax.set_ylabel('Price')
	plt.savefig("ohlc1.png")

	fig = plt.figure(figsize=(20,5))
	fig, ax = plt.subplots()
	candlestick_ohlc(ax, ohlc.values[-2:len(df.index.values.tolist())], width=0.4,colorup='g', colordown='r');
	ax.set_xlabel('Date')
	ax.set_ylabel('Price')
	plt.savefig("ohlc2.png")

	fig = plt.figure(figsize=(20,5))
	fig, ax = plt.subplots()
	candlestick_ohlc(ax, ohlc.values[-3:len(df.index.values.tolist())], width=0.4,colorup='g', colordown='r');
	ax.set_xlabel('Date')
	ax.set_ylabel('Price')
	plt.savefig("ohlc3.png")
	
	plt.close('all')


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
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
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
