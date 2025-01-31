#building the model

  model=Sequential()

  model.add(LSTM(units=50,return_sequence=True,input_shape=(x_train.shape[1],1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50,return_sequence=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50))
  model.add(Dropout(0.2))
  model.add(Dense(units=1)) # This predicts the next closing value of the stock

  model.compile(optimizer='adam',loss='mean_squared_error')
  model.fit(x_train,y_train,epochs=25,batch_size=32)