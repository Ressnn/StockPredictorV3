# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:28:28 2018

@author: ASUS
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
tickers  = ['NVDA']
start_date = '2015-01-01'
end_date = '2018-11-20'
target_size = 750
splits = 5

#Model settings//Change if you know what you are doing
test_list_size = 1
epochs = int(100)
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
model = Sequential()


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
shape = Batched[0][0].shape
new_shape = (1,shape[0],shape[1])
for i in range(0,epochs):
    model.fit(np.array(Batched[0][0]).reshape(new_shape),np.array(Batched[0][0]).reshape(new_shape))
# In[]
model.load_weights('ModelW.h5')

# In[]
ai_see = model.predict(np.array(Batched[0][0]).reshape(new_shape))
ai_predict = model.predict(np.array(Batched[0][1]).reshape(new_shape))
ai_see = pd.DataFrame(ai_see[0]).transpose()
ai_predict = pd.DataFrame(ai_predict[0]).transpose()
ai_see = ai_see.iloc[2,:].tolist()
ai_predict = ai_predict.iloc[2,:].tolist()
batched_1 = Batched[0][0][2].values.tolist()
batched_2 = Batched[0][1][2].values.tolist()
# In[]
dc = 2
import PredictionUtils
batched_2 = PredictionUtils.UndoDiff(batched_2)
ar_see = []
for i in ai_see:
    ar_see.append(i/3.5)
ar_pred = []
for i in ai_predict:
    ar_pred.append(i/3.5)
ai_see = PredictionUtils.UndoDiff(ar_see)
ai_predict = PredictionUtils.UndoDiff(ar_pred)
plt.plot(batched_2)
plt.plot(ai_see)
ai_p_s = []
for i in ai_predict:
    ai_p_s.append(i+batched_2[-1])

range_list = []
for i in range(int(target_size-(target_size/splits)),target_size):
    for a in range(0,splits-2):
        i = i-(target_size/splits)
    range_list.append(i)
    
plt.plot(range_list,ai_p_s)