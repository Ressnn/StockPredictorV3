# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 19:57:05 2018

@author: Pranav Devarinti
"""

import DataGetter
import DataShaper
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,CuDNNLSTM,CuDNNGRU,Dropout
from keras.activations import softsign,selu,elu,relu
#Variables go here

#Shifting around variables here is reccomended
n = ['QCOM','SYMC','NVDA','AMAT','PYPL','XLNX','TXN','WDC','STX','ADBE','MCHP','SWKS','FTNT','AVGO','ADSK','FISV','ADP','LRCX','CTSH','KLAC','ADI','AKAM','CDNS','PAYX','CTXS','QRVO','VRSN','FLIR','INTU','SNPS','ANSS','IPGP','V','HPE','ORCL','IBM','DXC','HPQ','MA']
tickers  = ['AMD','INTC','QCOM','TXN','NVDA']
start_date = '2017-01-01'
end_date = '2018-11-01'
target_size = 480
splits = 3
reps = 400
test_list_size = 1
epochs = int(1)
#Model settings//Change if you know what you are doing
eb = int(6)
nsub = int(target_size/splits)
import matplotlib.pyplot as plt

# In[]
class Data():
    stock_data = []
    trend_data = []
    stock_data, trend_data = DataGetter.Get_Data(tickers,start_date,end_date)

# In[]
# The market's close value is the third one to iloc
x = Data.stock_data
r = Data.trend_data
# In[]
nx = []
nr = []
final = []
for i in x:
    nx.append(DataShaper.Reshape2(i,target_size))
for i in r:
    nr.append(DataShaper.Reshape2(i,target_size))
for i in range(0,len(tickers)):
    final.append(np.vstack((np.array(nx[i]),np.array(nr[i]))))
# In[]
final = np.array(final)[:,0:6]
# In[]
Diffrenced = DataShaper.ProduceDiff(final)
Batched = DataShaper.Makebatches(Diffrenced,splits)
# In[]
model = Sequential()
try:
    model.add(CuDNNLSTM(1000,return_sequences=True))
    model.add(Dropout(.1))
    model.add(CuDNNLSTM(500,return_sequences=True))
    model.add(Dropout(.1))
    model.add(CuDNNLSTM(300,return_sequences=True))
    model.add(Dropout(.1))
    model.add(Dense(500,activation='selu'))
    model.add(Dense(300))
    model.add(Dropout(.1))
    model.add(Dense(100))
    model.add(Dense(6))
except:
    print('cudnn load failed!')
    model.add(LSTM(1000,return_sequences=True))
    model.add(LSTM(500,return_sequences=True))
    model.add(LSTM(300,return_sequences=True))
    model.add(Dropout(.3))
    model.add(Dense(500))
    model.add(Dense(300))
    model.add(Dense(6))
model.compile(loss='mse', optimizer='adam')
# In[]
Train_list = []
for i in Batched:
    for a in range(0,splits-1):
        Train_list.append([i[a],i[a+1]])
# In[]
for a in range(0,reps):
    for i in range(0,len(Train_list)-test_list_size):
        x = np.array(Train_list[i][0]).reshape(-1,int(target_size/splits),eb)
        y = np.array(Train_list[i][1]).reshape(-1,int(target_size/splits),eb)
        model.fit(x,y,epochs=epochs)
   # In[]
test_list = Train_list[-test_list_size:]
pred_list = []
r = []
for i in test_list:
    pred_list.append(model.predict(np.array(i[0]).reshape(1,nsub,eb)))
    r = np.transpose(np.array(i[0]))[4]
# In[]
showpredict = []
showpredict = np.transpose(pred_list[-1][0])
showonplot = showpredict[4].reshape(-1)
plt.plot(showonplot)
plt.plot(r)

# In[]
model.save_weights('ModelW.h5')
