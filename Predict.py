# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:28:28 2019

@author: ASUS
"""
# In[]
#All of Our Other File Imports - For Organization because the data can be tinkered with faster this way
import DataGetter
import DataShaper

import numpy as np
#Numpy: for general number processing and for the np.array datatype which is easy to index
import pandas as pd
#Pandas: for data processing
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,CuDNNLSTM,CuDNNGRU,Dropout,Flatten
from keras.activations import softsign,selu,elu,relu
import matplotlib.pyplot as plt
#Keras:to make and train NN

# In[]

#Variables go here
#Shifting around variables here is reccomended
tickers  = [str(input("Ticker to Predict? "))]
#When whould the date of input be?




start_date = '2017-08-10'
end_date = str(input("When should the predictions start? yy-mm-dd "))
target_size = 400
splits = 10



#Model settings//Change if you know what you are doing
test_list_size = 1
epochs = int(100)
eb = int(6)
nsub = int(target_size/splits)


# In[]
stock_data = []
trend_data = []
x, r = DataGetter.Get_Data(tickers,start_date,end_date)
# The market's close value is the third one to iloc

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
shape = Batched[0][0].shape
new_shape = (1,shape[0],shape[1])
for i in range(0,epochs):
    model.fit(np.array(Batched[0][0]).reshape(new_shape),np.array(Batched[0][0]).reshape(new_shape))
# In[]
model.load_weights('ModelW2.h5')

# In[]

sn0 = -1
sn1 = (-nsub*1)-1
sn2 = (-nsub*2)-1



ai_see = model.predict(np.array(x[0])[:,sn2:sn1].reshape(new_shape))
ai_predict = model.predict(np.array(x[0])[:,sn1:sn0].reshape(new_shape))
ai_see = pd.DataFrame(ai_see[0]).transpose()
ai_predict = pd.DataFrame(ai_predict[0]).transpose()
ai_see = ai_see.iloc[3,:].tolist()
ai_predict = ai_predict.iloc[3,:].tolist()
batched_1 = np.transpose(np.array(x[0])[:,sn2:sn1]).reshape(new_shape)
batched_2 = np.transpose(np.array(x[0])[:,sn1:sn0]).reshape(new_shape)
# In[]
dc = 2
import PredictionUtils
batched_2 
ar_see = []
for i in ai_see:
    ar_see.append(i)
ar_pred = []
for i in ai_predict:
    ar_pred.append(i)
ai_see = PredictionUtils.UndoDiff(ar_see)
ai_predict = PredictionUtils.UndoDiff(ar_pred)


add = np.array(batched_2)[0,-1 ,3]

ai_p_s = []
for i in ai_predict:
    ai_p_s.append(i)

range_list = []
for i in range(int(target_size-(target_size/splits)),target_size):
    for a in range(0,splits-2):
        i = i-(target_size/splits)
    range_list.append(i)
# In[]
plt.plot(np.array(batched_2)[0,:,3])
plt.plot(np.array(DataShaper.UndoDiff(ar_see)))

final_pred_list = np.array(ar_pred)
final_amount = (final_pred_list-np.mean(final_pred_list))/np.std(final_pred_list)
dt = DataShaper.UndoDiff(final_amount)
plt.plot(range_list,dt)
plt.plot(range_list,DataShaper.UndoDiff(final_pred_list))
# In[]
print("One Day Prediction : (Normal:"+ str(final_pred_list[0])+" Standardized:"+ str(final_amount[0])+")")
print("Two Day Prediction : (Normal:"+ str(final_pred_list[1])+" Standardized:"+ str(final_amount[1])+")")

