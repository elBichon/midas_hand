from datetime import datetime
from iexfinance.stocks import get_historical_intraday
from iexfinance.stocks import Stock
import plotly.graph_objects as go
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
#import numpy as np
import itertools
# to not display the warnings of tensorflow
import os
import utils
import time
from mpl_finance import candlestick_ohlc
import credentials 
plt.style.use('ggplot')


x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []




while True:
	df = get_historical_intraday("CSCO", output_format='pandas',token= credentials.token)
	df = df.fillna(method='ffill')

	fft_100_high = []
	fft_100_low = []
	fft_100_close = []
	fft_100_open = []
	fft_20_high = []
	fft_20_low = []
	fft_20_close = []
	fft_20_open = []

	df_close = df['close'].values.tolist()
	df_fft_close = utils.fourier_transform(df,'close',fft_20_close,fft_100_close)
	df_fft_low = utils.fourier_transform(df,'low',fft_20_low,fft_100_low)
	df_fft_open = utils.fourier_transform(df,'open',fft_20_open,fft_100_open)
	df_fft_open = utils.fourier_transform(df,'high',fft_20_high,fft_100_high)
	volume = df['volume'].values.tolist()
	fft_20_close = list(itertools.chain.from_iterable(fft_20_close))
	fft_20_high = list(itertools.chain.from_iterable(fft_20_high))
	fft_20_low = list(itertools.chain.from_iterable(fft_20_low))
	fft_20_open = list(itertools.chain.from_iterable(fft_20_open))
	fft_100_close = list(itertools.chain.from_iterable(fft_100_close))
	fft_100_high = list(itertools.chain.from_iterable(fft_100_high))
	fft_100_low = list(itertools.chain.from_iterable(fft_100_low))
	fft_100_open = list(itertools.chain.from_iterable(fft_100_open))

	df_dict = {'index': len(list(range(0,len(df)))), 'date':df.date.values.tolist(),'close':df_close,'fft_20_close':fft_20_close,'fft_20_low':fft_20_low,'fft_20_high':fft_20_high,'fft_20_open':fft_20_open,'fft_100_close':fft_100_close,'fft_100_high':fft_100_high,'fft_100_low':fft_100_low,'fft_100_open':fft_100_open,'volume':volume}
	df = pd.DataFrame(df_dict)

	df = df[['date','index','close','fft_20_close','fft_20_low','fft_20_high','fft_20_open','fft_100_close','fft_100_high','fft_100_open','fft_100_low']]
	df = utils.get_technical_indicators(df)

	fft_20_close = df.fft_20_close.values.tolist()
	fft_100_close = df.fft_100_close.values.tolist()

	stoch_indicator = utils.stoch_indicator(df['%K'].values.tolist(),df['%D'].values.tolist())
	bollinger_indicator = utils.bollinger_indicator(df['upper_band'].values.tolist(),df['lower_band'].values.tolist(),fft_20_close)
	df['RSI'] = utils.computeRSI(df['fft_20_close'], 14)
	rsi_list = df['RSI'].values.tolist()
	rsi_indicator = utils.rsi_indicator(df['RSI'].values.tolist())
	ema_indicator = utils.ema_indicator(df['12ema'].values.tolist(),df['26ema'].values.tolist(),df.index.values.tolist(),fft_20_close)

	df_dict = {'index':df.index.values.tolist(),'date':df.date.values.tolist(),'volume':volume,'close':df.close.values.tolist(),'fft_20_close': df.fft_20_close.values.tolist(),'fft_20_open': df.fft_20_open.values.tolist(),'fft_20_low': df.fft_20_low.values.tolist(),'fft_20_high': df.fft_20_high.values.tolist(),'fft_100_close': df.fft_100_close.values.tolist(),'fft_100_low': df.fft_100_low.values.tolist(),'fft_100_high': df.fft_100_high.values.tolist(),'fft_100_open': df.fft_100_open.values.tolist(),'fft_100_close': df.fft_100_close.values.tolist(),'fft_100_low': df.fft_100_low.values.tolist(),'fft_100_high': df.fft_100_high.values.tolist(),'fft_100_open': df.fft_100_open.values.tolist(),'rsi_indicator':rsi_indicator,'stoch_indicator':stoch_indicator,'%K':df['%K'].values.tolist(),'%D':df['%D'].values.tolist(),'rsi':df['RSI'].values.tolist(),'ma20':df['ma20'].values.tolist(),'ma50':df['ma50'].values.tolist(),'26ema':df['26ema'].values.tolist(),'12ema':df['12ema'].values.tolist(),'upper_band':df['upper_band'].values.tolist(),'lower_band':df['lower_band'].values.tolist(),'ema_indicator':ema_indicator,'bollinger_indicator':bollinger_indicator,'var_ema':df.var_ema.values.tolist(),'var_bollinger':df.var_bollinger.values.tolist()}
	df = pd.DataFrame(df_dict)

	rsi = df.rsi_indicator.values.tolist()
	stoch = df.stoch_indicator.values.tolist()
	bollinger = df.bollinger_indicator.values.tolist()
	ema_12 = df['12ema'].values.tolist()
	ema_26 = df['26ema'].values.tolist()
	index = df.index.values.tolist()

	utils.ema_trade_indicator(ema_12,ema_26,index,fft_20_close,x2,y2)
	utils.rsi_stoch_trade_indicator(index,rsi,stoch,fft_20_close,x1,y1)
	utils.bollinger_trade_indicator(bollinger,index,fft_20_close,x3,y3)
	
	df_dict = {'index':df.index.values.tolist(),'date':df.date.values.tolist(),'volume':volume,'close':df.close.values.tolist(),'fft_20_close': df.fft_20_close.values.tolist(),'fft_20_open': df.fft_20_open.values.tolist(),'fft_20_low': df.fft_20_low.values.tolist(),'fft_20_high': df.fft_20_high.values.tolist(),'fft_100_close': df.fft_100_close.values.tolist(),'fft_100_low': df.fft_100_low.values.tolist(),'fft_100_high': df.fft_100_high.values.tolist(),'fft_100_open': df.fft_100_open.values.tolist(),'fft_100_close': df.fft_100_close.values.tolist(),'fft_100_low': df.fft_100_low.values.tolist(),'fft_100_high': df.fft_100_high.values.tolist(),'fft_100_open': df.fft_100_open.values.tolist(),'var_ema':df['var_ema'].values.tolist(),'var_bollinger':df.var_bollinger.values.tolist(),'rsi_indicator':rsi_indicator,'stoch_indicator':stoch_indicator,'%K':df['%K'].values.tolist(),'%D':df['%D'].values.tolist(),'RSI':rsi_list,'ma20':df['ma20'].values.tolist(),'ma50':df['ma50'].values.tolist(),'26ema':df['26ema'].values.tolist(),'12ema':df['12ema'].values.tolist(),'upper_band':df['upper_band'].values.tolist(),'lower_band':df['lower_band'].values.tolist(),'ema_indicator':ema_indicator,'bollinger_indicator':bollinger_indicator}
	df = pd.DataFrame(df_dict)


	utils.generate_graph(df,x1,y1,x2,y2,x3,y3)
	print(df.tail())
	time.sleep(60)
