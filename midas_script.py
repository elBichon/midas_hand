import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from subprocess import check_output
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# to not display the warnings of tensorflow
import os
import utils



stock_name = 'AF'  #AF   AIR   AI
stock = stock_name+'.csv'

fft_3_high = []
fft_6_high = []
fft_10_high = []
fft_20_high = []
fft_100_high = []

fft_3_low = []
fft_6_low = []
fft_10_low = []
fft_20_low = []
fft_100_low = []

fft_3_close = []
fft_6_close = []
fft_10_close = []
fft_20_close = []
fft_100_close = []

df = utils.format_dataset(stock)

df_fft_close = utils.fourier_transform(df,'close',fft_3_close,fft_6_close,fft_10_close,fft_20_close,fft_100_close)
df_fft_low = utils.fourier_transform(df,'low',fft_3_low,fft_6_low,fft_10_low,fft_20_low, fft_100_low)
df_fft_high = utils.fourier_transform(df,'high',fft_3_high,fft_6_high,fft_10_high,fft_20_high,fft_100_high)

fft_3_close = list(itertools.chain.from_iterable(fft_3_close))
fft_6_close = list(itertools.chain.from_iterable(fft_6_close))
fft_10_close = list(itertools.chain.from_iterable(fft_10_close))
fft_20_close = list(itertools.chain.from_iterable(fft_20_close))
fft_100_close = list(itertools.chain.from_iterable(fft_100_close))

fft_3_high = list(itertools.chain.from_iterable(fft_3_high))
fft_6_high = list(itertools.chain.from_iterable(fft_6_high))
fft_10_high = list(itertools.chain.from_iterable(fft_10_high))
fft_20_high = list(itertools.chain.from_iterable(fft_20_high))
fft_100_high = list(itertools.chain.from_iterable(fft_100_high))

fft_3_low = list(itertools.chain.from_iterable(fft_3_low))
fft_6_low = list(itertools.chain.from_iterable(fft_6_low))
fft_10_low = list(itertools.chain.from_iterable(fft_10_low))
fft_20_low = list(itertools.chain.from_iterable(fft_20_low))
fft_100_low = list(itertools.chain.from_iterable(fft_100_low))



df_dict = {'index': list(range(0,len(df))),'name':df.name.values.tolist(), 'date':df.date.values.tolist(), 'volume':df.volume.values.tolist(), 'fft_3_close':fft_3_close,'fft_6_close':fft_6_close,'fft_10_close':fft_10_close, 'fft_20_close':fft_20_close, 'fft_100_close':fft_100_close, 'fft_3_high':fft_3_high,'fft_6_high':fft_6_high,'fft_10_high':fft_10_high,'fft_20_high':fft_20_high,'fft_100_high':fft_100_high,'fft_3_low':fft_3_low , 'fft_6_low':fft_6_low,'fft_10_low':fft_10_low,'fft_20_low':fft_20_low,'fft_100_low':fft_100_low}
df = pd.DataFrame(df_dict) 
df = utils.get_technical_indicators(df,'fft_100_close')
df2 = utils.get_technical_indicators(df,'fft_100_close')
print(df2.head())
df = df[['date', 'fft_3_close','fft_6_close','fft_10_close','fft_100_close', 'fft_3_low','fft_6_low','fft_10_low','fft_20_low', 'fft_100_low', 'fft_3_high','fft_6_high','fft_10_high','fft_20_high','fft_100_high', 'volume', 'var_mma', 'var_macd','ema']]#'var_bollinger', 'ema'#,'upper_band', 'lower_band']]
pct_close_mvt = utils.get_movement(df,fft_100_close)
fft = df.fft_100_close.values.tolist()
fft_dict = {'fft':fft}
fft = pd.DataFrame(fft_dict) 
#momentum = []
#momentum = utils.generate_momentum(fft,momentum)
df['%K'] = utils.STOK(df['fft_100_close'], df['fft_100_low'], df['fft_100_high'], 14)
df['%D'] = utils.STOD(df['fft_100_close'], df['fft_100_low'], df['fft_100_high'], 14)

k_list = df['%K'].values.tolist()
k_indicator = []
k_value = []

d_list = df['%D'].values.tolist()
d_indicator = []
d_value = []

fft20_low = df.fft_20_low.values.tolist()
fft20_high = df.fft_20_high.values.tolist()
variation = [0]
variation_low = [0]
variation_high = [0]

i = 1
while i < len(fft20_low):
    if fft20_low[i]-fft20_low[i-1] > 0:
        variation_low.append(1)
    else:
        variation_low.append(0)
    i += 1

i = 1
while i < len(fft20_high):
    if fft20_high[i]-fft20_high[i-1] > 0:
        variation_high.append(1)
    else:
        variation_high.append(0)
    i += 1

i = 0
while i < len(variation_high):
    if variation_low[i] == 0 and variation_high[i] == 0:
        variation.append(-1)
    elif variation_low[i] == 1 and variation_high[i] == 1:
        variation.append(1)
    else:
        variation.append(0)
    i += 1

print('==========')
print('variation')
print(variation)
print('==========')

while i < len(k_list):
    if k_list[i] > 80:
        k_indicator.append(1)
        k_value.append(k_list[i])
    elif k_list[i] < 20:
        k_indicator.append(-1)
        k_value.append(k_list[i])
    else:
        k_indicator.append(0)
        k_value.append(0)
    if d_list[i] > 80:
        d_indicator.append(1)
        k_value.append(d_list[i])
    elif d_list[i] < 20:
        d_indicator.append(-1)
        k_value.append(d_list[i])
    else:
        d_indicator.append(0)
        k_value.append(0)
    i += 1


df.plot(y=['fft_100_close'], figsize = (20, 5))
df.plot(y=['%K', '%D'], figsize = (20, 5))

df['RSI'] = utils.computeRSI(fft['fft'], 14)

rsi_list = df['RSI'].values.tolist()
rsi_indicator = []
i = 0
while i < len(rsi_list):
    if rsi_list[i] > 80:
        rsi_indicator.append(1)
    elif rsi_list[i] < 20:
        rsi_indicator.append(-1)
    else:
        rsi_indicator.append(0)
    i += 1




# plot correspondingRSI values and significant levels
plt.figure(figsize=(15,5))
plt.title('RSI chart')
plt.plot(df['date'], df['RSI'])

plt.axhline(0, linestyle='--', alpha=0.1)
plt.axhline(20, linestyle='--', alpha=0.5)
plt.axhline(30, linestyle='--')

plt.axhline(70, linestyle='--')
plt.axhline(80, linestyle='--', alpha=0.5)
plt.axhline(100, linestyle='--', alpha=0.1)
plt.show()

i = 0
#var_bollinger = df.var_bollinger.values.tolist()

#variation = []
#i = 1
#while i < len(var_bollinger):
#    variation.append(var_bollinger[i]-var_bollinger[i-1])
#    i += 1

#upper = df.upper_band.values.tolist()
#lower = df.lower_band.values.tolist()
#stock_bollinger_upper = []
#stock_bollinger_lower = []
#value_bollinger = []
#i = 0
#while i < len(fft):
#    stock_bollinger_upper.append(fft[i]-upper[i])
#    stock_bollinger_lower.append(fft[i]-lower[i])
#    i += 1
    
#i = 0
#while i < len(fft):
#    if str(stock_bollinger_upper[i]) == 'nan':
#        value_bollinger.append(100)
#    elif stock_bollinger_upper[i] > 0:
#        value_bollinger.append(1)
#    elif stock_bollinger_lower[i] < 0:
#        value_bollinger.append(-1)
#    elif stock_bollinger_upper[i] < 0 and  stock_bollinger_lower[i] > 0:
#        value_bollinger.append(0)
#    i += 1

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

df_dict = {'date':df.date.values.tolist(),'pct_close_mvt':pct_close_mvt,'fft_3_close':df.fft_3_close.values.tolist(),'fft_6_close':df.fft_6_close.values.tolist(),'fft_10_close':df.fft_10_close.values.tolist(),'fft_3_low':df.fft_3_low.values.tolist(),'fft_6_low':df.fft_6_low.values.tolist(),'fft_10_low':df.fft_10_low.values.tolist(),'fft_3_high':df.fft_3_high.values.tolist(),'fft_6_high': df.fft_6_high.values.tolist(),'fft_10_high':df.fft_10_high.values.tolist(),'fft_20_low': df.fft_20_low.values.tolist(),'fft_20_high': df.fft_20_high.values.tolist(),'fft_100_close': df.fft_100_close.values.tolist(),'fft_100_low': df.fft_100_low.values.tolist(),'fft_100_high': df.fft_100_high.values.tolist(),'value_macd':value_macd,'value_mma':value_mma,'variation':variation,'rsi_indicator':rsi_indicator}
df = pd.DataFrame(df_dict) 

print(df.head())
#momentum = df.momentum.values.tolist()

#label = []
#i = 0
#while i < len(momentum)-1:
#    if momentum[i] == 0 and momentum[i+1] == 0:
#        label.append(0)
#    elif momentum[i] == 1 and momentum[i+1] == 1:
#        label.append(0)
#    elif momentum[i] == 1 and momentum[i+1] == 0:
#        label.append(1)
#    elif momentum[i] == 0 and momentum[i+1] == 1:
#        label.append(2)
#    i += 1
#if momentum[241] == 0 and momentum[242] == 0:
#    label.append(0)
#elif momentum[241] == 1 and momentum[242] == 1:
#    label.append(0)
#elif momentum[241] == 0 and momentum[242] == 1:
#    label.append(1)
#elif momentum[241] == 1 and momentum[242] == 0:
#    label.append(2)

print('==========')
print('==========')
print(df.columns)
plt.figure(figsize=(20,10))
plt.plot(df.date.values.tolist(),df.fft_100_close.values.tolist(), label='fft100_close')
plt.plot(df.date.values.tolist(),df2.ma20.values.tolist(), label='ma20')
plt.plot(df.date.values.tolist(),df2.ma50.values.tolist(), label='ma50')

plt.plot(df.date.values.tolist(),df2.amd20.values.tolist(), label='ema20')
plt.plot(df.date.values.tolist(),df2.amd50.values.tolist(), label='ema50')
#plt.plot(df.date.values.tolist(),df.upper_band.values.tolist(), label='upper')
#plt.plot(df.date.values.tolist(),df.lower_band.values.tolist(), label='lower')
plt.legend()
plt.show()

#plt.figure(figsize=(20,10))
#plt.plot(df.date.values.tolist(),df.MACD.values.tolist(), label='MACD')
#plt.plot(df.date.values.tolist(),df.signal.values.tolist(), label='signal')
#plt.legend()
#plt.show()

plt.figure(figsize=(20,10))
plt.plot(df.date.values.tolist(),df.fft_100_close.values.tolist(), label='fft_100_close')
plt.plot(df.date.values.tolist(),df.fft_20_high.values.tolist(), label='fft_20_high')
plt.plot(df.date.values.tolist(),df.fft_20_low.values.tolist(), label='fft_20_low')
plt.plot(df.date.values.tolist(),df.fft_6_close.values.tolist(), label='fft_6_close')
plt.plot(df.date.values.tolist(),df.fft_3_close.values.tolist(), label='fft_3_close')
plt.legend()
plt.show()




