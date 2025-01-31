import yfinance as yf
import matplotlib.pyplot as pt
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten
from keras.models import load_model



company='reliance.ns'
time='48mo'

data=yf.Ticker(company).history(period=time) # this api call is used to retreive the stock data 

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data['Close'].values.reshape(-1,1))  # used to transform the data in the scale of 0 to 1 
  

prediction_days=100      #  It has to be lesser than the time mentioned above
x_train=[]
y_train=[]

for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

x_train,y_train=np.array(x_train),np.array(y_train)




model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1],1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

model.save('conv2d_model1.h5')

