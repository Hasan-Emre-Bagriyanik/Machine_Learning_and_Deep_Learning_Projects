# AIRLINE PASSENGER (TIME SERIES) PREDICTION USING LSTM

# We will use Airline Passenger dataset for this project. This dataset provides monthly totals of a US airline passengers from 1949 to 1960. You can download the dataset from Kaggle link below: 
# https://www.kaggle.com/chirag19/air-passengers

# We will use LSTM deep learning model for this project. The Long Short-Term Memory network, or LSTM network, is a recurrent neural network that is trained using Backpropagation through time and 
# overcomes the vanishing gradient problem. LSTM can be used to create large recurrent networks that in turn can be used to address difficult sequence problems in machine learning and achieve 
# state-of-the-art results. Instead of neurons, LSTM networks have memory blocks that are connected through layers.

#  Aim of the project:
# Given the number of passengers (in units of thousands) for last two months, what is the number of passengers next month? In order to solve this problem we will build a LSTM model and train 
# this model with our train data which is first 100 months in our dataset. After the LSTM model training finishes and learn the pattern in time series train data, we will ask it the above question  
# question and get the answer from it.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# You can downlad the dataset from Kaggle link below:
# https://www.kaggle.com/chirag19/air-passengers
data = pd.read_csv("AirPassengers.csv")
data.head()

data.rename(columns = {"#Passengers": "passengers"}, inplace = True)
# Since this is a time series, we need only second column.. So data now contains only passenger count...
data = data["passengers"]

type(data)

# My data fromat is Series, but I need 2D array for MinMaxScaler() and my other methods to work. So I will change to numpy array and reshape it.
data = np.array(data).reshape(-1,1)

# ok, now we have 2D numpy array...
type(data)

plt.plot(data)
plt.show()
#%%
# LSTM is sensitive to the scale of the input data. So we will rescale the data to the range of 0-to-1, also called normalizing. 
# scaling
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# ### Train, Test split
len(data)

# I have 144 data. I will use 100 of it as train set and 44 as test set..

train = data[0:100,:]
test = data[100:,:]

#%%
# We will now define a function to prepare the train and test datasets for modeling. The function takes two arguments: the dataset, 
# which is a NumPy array that we want to convert into a dataset, and the steps, which is the number of previous time steps to use as input variables to predict the next time period.

def get_data(data,steps):
    dataX = []
    dataY = []
    for i in range(len(data) - steps -1):
        a = data[i:(i+steps), 0]
        dataX.append(a)
        dataY.append(data[i+steps, 0])
    return np.array(dataX), np.array((dataY))

# So using this "get_data" function I will prepare a dataset for modeling... Then I give this new prepared datset to my model for training...

steps = 3

# #### Now I'm making my datasets for both training and testing..

# Important: You must have numpy version 1.19 in your Anaconda environment for LSTM work. If you have a error like "NotImplementedError: 
# Cannot convert a symbolic Tensor (lstm/strided_slice:0) to a numpy array." you must change your numpy version to 1.19 using this commnad:
# conda install numpy=1.19

x_train, y_train = get_data(train, steps)
x_test, y_test = get_data(test, steps)


# Im reshaping my sets for using in LSTM model..
x_train = np.reshape(x_train,(x_train.shape[0], 1 ,x_train.shape[1]))
x_test = np.reshape(x_test,(x_test.shape[0], 1, x_test.shape[1]))

#%%
# I will use a Sequential model with 2 hidden layers
# Instead of neurons, LSTM networks have memory blocks that are connected through layers.
# The default sigmoid activation function is used for the LSTM blocks. 

model = Sequential()
model.add(LSTM(128, input_shape = (1,steps)))
model.add(Dense(64))
model.add(Dense(1))   # This is my output layer
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.summary()

# ### Now it's time to train our model...
model.fit(x_train, y_train, epochs=50, batch_size=1)


#%%
y_pred = model.predict(x_test)

# We should rescale the prediction results, because our model gives us scaled predictions..
y_pred = scaler.inverse_transform(y_pred)
y_test = y_test.reshape(-1,1)
y_test = scaler.inverse_transform(y_test)

#  Now plot the test set results... Remember our test set contains last 44 data in original dataset..

# plot real number of passengers and predictions...
plt.plot(y_test, label = "real number of passengers")
plt.plot(y_pred, label = "predicted number of passengers")
plt.ylabel("Months")
plt.xlabel("Number of passengers")
plt.legend()
plt.show()
