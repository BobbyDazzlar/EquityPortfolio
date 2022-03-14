
import pandas as pd
import numpy as np

import numpy

#LSTM scores
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#LSTM
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

#Neuron Parameters
from kerastuner.tuners import RandomSearch



x=pd.read_csv('n50.csv',parse_dates=['Date'],index_col='Date')
x = x.loc["2016-01-01" :]                         #Since 2016-01-01, 5y(1234rows till 2020-12-31), + year 2021's rows (till 30th of April)
y=x.copy()                                        #deep copy
x.reset_index(drop=True, inplace=True)


def create_dataset(dataset, time_step=1):         # convert an array of values into a dataset matrix which will be used to train the lstm model.
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step):
		a = dataset[i:(i+time_step), 0]               #i=0, 0,1,2,3-----(timesteps-1)  -> timesteps
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(x).reshape(-1,1))
after2020=len(y.loc["2021-01-01" : ])
before_2021_data_length=int(len(df1)-after2020)                 #length of data before 2021
training_size=int(before_2021_data_length)
train_data=df1[0:training_size,:]
test_data=df1[after2020:,:1]
inpdata=df1[before_2021_data_length-60:len(df1),:1]


X_train, y_train = create_dataset(train_data, 60)
x_inp, y_inp = create_dataset(inpdata, 60)
x_test, y_test = create_dataset(test_data,60)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
x_inp = x_inp.reshape(x_inp.shape[0],x_inp.shape[1] , 1)        #reshape input to be [samples, time steps, features] which is required for LSTM
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

def build_model(hp):
  # initialising stacked lstm
  model=Sequential()
  for i in range(hp.Int('num_layers', 2, 7)):
    model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=8),
                               activation='tanh'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error',
        metrics=['mean_squared_error'])
  return model

tuner=RandomSearch(
    build_model,
    objective='mean_squared_error',
    max_trials=5,
    executions_per_trial=3,
    directory='LayerNeurons',
    project_name='parameters')

tuner.search(X_train, y_train,epochs=100,validation_data=(x_test, y_test))

print(tuner.results_summary())