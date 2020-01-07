# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:28:28 2019

@author: Pranav Devarinti
"""

import DataGetter
import DataShaper
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,CuDNNLSTM,CuDNNGRU,Dropout,Flatten
from keras.activations import softsign,selu,elu,relu
from sklearn.model_selection import KFold
#Variables go here

#Shifting around variables here is reccomended
n = ['QCOM','SYMC','NVDA','AMAT','PYPL','XLNX','TXN','WDC','STX','ADBE','MCHP','SWKS','FTNT','AVGO','ADSK','FISV','ADP','LRCX','CTSH','KLAC','ADI','AKAM','CDNS','PAYX','CTXS','QRVO','VRSN','FLIR','INTU','SNPS','ANSS','IPGP','V','HPE','ORCL','IBM','DXC','HPQ','MA']
tickers  = ['QCOM','SYMC','NVDA','AMAT','PYPL','XLNX','TXN','WDC','STX','ADBE','MCHP','SWKS','FTNT','AVGO','ADSK','FISV','ADP','LRCX','CTSH','KLAC','ADI','AKAM','CDNS','PAYX','CTXS','QRVO','VRSN','FLIR','INTU','SNPS','ANSS','IPGP','V','HPE','ORCL','IBM','DXC','HPQ','MA']
start_date = '2014-01-01'
end_date = '2019-10-01'
target_size = 300
splits = 20
reps = 1000
test_list_size = 1
batch_size = 8
epochs = int(5)
stp = 1000
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
    model.add(CuDNNLSTM(75,return_sequences=True))
    model.add(CuDNNLSTM(40,return_sequences=True))
    model.add(Dense(125,activation='tanh'))
    model.add(Dense(45))
    model.add(Dropout(.2))
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
        '''
for a in range(0,reps):
    for i in range(0,len(Train_list)-test_list_size):
        x = np.array(Train_list[i][0]).reshape(-1,int(target_size/splits),eb)
        y = np.array(Train_list[i][1]).reshape(-1,int(target_size/splits),eb)
        model.fit(x,y,epochs=epochs,batch_size=batch_size)
        '''
# In[]
xl = []
yl = []
for a in range(0,reps):
    for i in range(0,len(Train_list)-test_list_size):
        xl.append(np.array(Train_list[i][0]).reshape(int(target_size/splits),eb))
        yl.append(np.array(Train_list[i][1]).reshape(int(target_size/splits),eb))
# In[]
kfold = KFold(10, True, 1)
data = []
for i in range(len(xl)):
    data.append(i)
    
for train, test in kfold.split(data):

    trainx = []
    trainy = []
    testx = []
    testy =[]
    
    for i in train:
        trainx.append(xl[i])
        trainy.append(yl[i])
    for i in test:
        testx.append(xl[i])
        testy.append(yl[i])
    
    model.fit(np.array(trainx),np.array(trainy),epochs=3,batch_size=batch_size,validation_data=(np.array(testx),np.array(testy)))
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
