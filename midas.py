import datetime as dt
from matplotlib import style
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import utils


stock_name = 'AI'  #AF   AIR   AI
stock = stock_name+'.csv'

fft_3_high = []
fft_6_high = []
fft_10_high = []
fft_20_high = []
fft_50_high = []
fft_100_high = []
fft_3_low = []
fft_6_low = []
fft_10_low = []
fft_20_low = []
fft_50_low = []
fft_100_low = []
fft_3_close = []
fft_6_close = []
fft_10_close = []
fft_20_close = []
fft_50_close = []
fft_100_close = []
fft_3_open = []
fft_6_open = []
fft_10_open = []
fft_20_open = []
fft_50_open = []
fft_100_open = []

df = utils.format_dataset(stock)
df_fft_close = utils.fourier_transform(df,'close',fft_3_close,fft_6_close,fft_10_close,fft_20_close,fft_50_close,fft_100_close)
df_fft_low = utils.fourier_transform(df,'low',fft_3_low,fft_6_low,fft_10_low,fft_20_low,fft_50_low, fft_100_low)
df_fft_high = utils.fourier_transform(df,'high',fft_3_high,fft_6_high,fft_10_high,fft_20_high,fft_50_high,fft_100_high)
df_fft_open = utils.fourier_transform(df,'high',fft_3_open,fft_6_open,fft_10_open,fft_20_open,fft_50_open,fft_100_open)

fft_3_close = list(itertools.chain.from_iterable(fft_3_close))
fft_6_close = list(itertools.chain.from_iterable(fft_6_close))
fft_10_close = list(itertools.chain.from_iterable(fft_10_close))
fft_20_close = list(itertools.chain.from_iterable(fft_20_close))
fft_50_close = list(itertools.chain.from_iterable(fft_50_close))
fft_100_close = list(itertools.chain.from_iterable(fft_100_close))

fft_3_high = list(itertools.chain.from_iterable(fft_3_high))
fft_6_high = list(itertools.chain.from_iterable(fft_6_high))
fft_10_high = list(itertools.chain.from_iterable(fft_10_high))
fft_20_high = list(itertools.chain.from_iterable(fft_20_high))
fft_50_high = list(itertools.chain.from_iterable(fft_50_high))
fft_100_high = list(itertools.chain.from_iterable(fft_100_high))

fft_3_low = list(itertools.chain.from_iterable(fft_3_low))
fft_6_low = list(itertools.chain.from_iterable(fft_6_low))
fft_10_low = list(itertools.chain.from_iterable(fft_10_low))
fft_20_low = list(itertools.chain.from_iterable(fft_20_low))
fft_50_low = list(itertools.chain.from_iterable(fft_50_low))
fft_100_low = list(itertools.chain.from_iterable(fft_100_low))

fft_50_open = list(itertools.chain.from_iterable(fft_50_open))

df_dict = {'index': list(range(0,len(df))),'name':df.name.values.tolist(), 'date':df.date.values.tolist(),'volume':df.volume.values.tolist(),'fft_3_close':fft_3_close,
'fft_6_close':fft_6_close,'fft_10_close':fft_10_close, 'fft_20_close':fft_20_close, 'fft_100_close':fft_100_close,'fft_3_high':fft_3_high,'fft_6_high':fft_6_high,'fft_10_high':fft_10_high,'fft_20_high':fft_20_high,
'fft_100_high':fft_100_high,'fft_3_low':fft_3_low, 'fft_6_low':fft_6_low,'fft_10_low':fft_10_low,'fft_20_low':fft_20_low,'fft_100_low':fft_100_low,'fft_50_open':fft_50_open,'fft_50_low':fft_50_low,'fft_50_close':fft_50_close,'fft_50_high':fft_50_high}
df = pd.DataFrame(df_dict) 
df = utils.get_technical_indicators(df,'fft_100_close')
df2 = utils.get_technical_indicators(df,'fft_100_close')

df = df[['date', 'fft_50_open','fft_50_low','fft_50_close','fft_50_high','fft_3_close','fft_6_close','fft_10_close','fft_100_close', 'fft_3_low','fft_6_low','fft_10_low','fft_20_low', 'fft_100_low', 'fft_3_high','fft_6_high','fft_10_high','fft_20_high','fft_100_high', 'volume', 'var_mma', 'var_macd','ema']]

pct_close_mvt = utils.get_movement(df,fft_100_close)
fft = df.fft_100_close.values.tolist()
fft_dict = {'fft':fft}
fft = pd.DataFrame(fft_dict) 

df['%K'] = utils.STOK(df['fft_100_close'], df['fft_100_low'], df['fft_100_high'], 14)
df['%D'] = utils.STOD(df['fft_100_close'], df['fft_100_low'], df['fft_100_high'], 14)

k_list = df['%K'].values.tolist()
d_list = df['%D'].values.tolist()
k_indicator = []
k_value = []
d_indicator = []
d_value = []

fft20_low = df.fft_20_low.values.tolist()
fft20_high = df.fft_20_high.values.tolist()
var = []
variation_low = [0]
variation_high = [0]
variation = utils.trend_estimation(fft20_low,fft20_high,variation_low,variation_high,var)

#stochastic = utils.compute_stochastic(k_list,d_list,k_value,d_value)

df['RSI'] = utils.computeRSI(fft['fft'], 14)
rsi = []
rsi_list = df['RSI'].values.tolist()
rsi_indicator = utils.rsi_indicator(rsi_list,rsi)

macd = []
var_macd_list = df.var_macd.values.tolist()
value_macd = utils.compute_macd(var_macd_list,macd)

mma = []
var_mma_list = df.var_mma.values.tolist()
value_mma = utils.compute_mma(var_mma_list,mma)

df_dict = {'date':df.date.values.tolist(),'pct_close_mvt':pct_close_mvt,
'fft_20_low': df.fft_20_low.values.tolist(),'fft_20_high': df.fft_20_high.values.tolist(),'fft_50_open': df.fft_50_open.values.tolist(),'fft_50_low': df.fft_50_low.values.tolist(),'fft_50_close': df.fft_50_close.values.tolist(),'fft_50_high': df.fft_50_high.values.tolist(),'fft_100_close': df.fft_100_close.values.tolist(),'value_macd':value_macd,'value_mma':value_mma,'variation':variation,'rsi_indicator':rsi_indicator}
df = pd.DataFrame(df_dict) 
print(df.head(50))
print(df.tail(50))

#plt.figure(figsize=(20,10))
#plt.plot(df.date.values.tolist(),df.fft_100_close.values.tolist(), label='fft_100_close')
#plt.plot(df.date.values.tolist(),df2.ma20.values.tolist(), label='ma20')
#plt.plot(df.date.values.tolist(),df2.ma50.values.tolist(), label='ma50')
#plt.plot(df.date.values.tolist(),df2.amd20.values.tolist(), label='ema20')
#plt.plot(df.date.values.tolist(),df2.amd50.values.tolist(), label='ema50')
#plt.legend()
#plt.show()

#plt.figure(figsize=(20,10))
#plt.plot(df.date.values.tolist(),df.fft_100_close.values.tolist(), label='fft_100_close')
#plt.plot(df.date.values.tolist(),df.fft_20_high.values.tolist(), label='fft_20_high')
#plt.plot(df.date.values.tolist(),df.fft_20_low.values.tolist(), label='fft_20_low')
#plt.legend()
#plt.show()

plt.figure(figsize=(20,10))
plt.plot(df.date.values.tolist(),df.fft_50_open.values.tolist(), label='fft_50_open')
plt.plot(df.date.values.tolist(),df.fft_50_low.values.tolist(), label='fft_50_low')
plt.plot(df.date.values.tolist(),df.fft_50_close.values.tolist(), label='fft_50_close')
plt.plot(df.date.values.tolist(),df.fft_50_high.values.tolist(), label='fft_50_high')
plt.legend()
plt.show()




