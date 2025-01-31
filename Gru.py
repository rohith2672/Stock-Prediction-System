import yfinance as yf
import matplotlib.pyplot as pt
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,BatchNormalization,GRU



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
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
 
  #building the model

model=Sequential()
model.add(GRU(64, input_shape=(28, 28)))
model.add(BatchNormalization())
model.add(Dense(10))


# preparing the testing data

test_data=yf.Ticker(company).history(period=time)
actual_prices=test_data['Close'].values
total_dataset=pd.concat((data['Close'],test_data['Close']),axis=0)

model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs=model_inputs.reshape(-1,1)
model_inputs=scaler.transform(model_inputs)

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)	

model.fit(
    x_train, y_train, validation_data=(x_train,y_train), batch_size=64, epochs=10
)

x_test=[]

for x in range (prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_prices=model.predict(x_test)
predicted_prices=scaler.inverse_transform(predicted_prices)




pt.plot(actual_prices,color="blue",label=f"Actual {company} Price")
pt.plot(predicted_prices,color="red",label=f"Predicted {company} Price")
pt.title(f"{company} Share price")
pt.xlabel('Time')
pt.ylabel(f"{company}Share Price")
pt.legend()
pt.show()