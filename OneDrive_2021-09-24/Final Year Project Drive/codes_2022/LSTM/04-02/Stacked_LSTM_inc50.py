import pandas as pd
import numpy as np
import numpy
import optuna
from numpy import array
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM



#code for MAPE, referred from the url: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
#'eps' is an arbitrary small yet strictly positive number to avoid undefined results when y is zero.
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps=0.01
    for i in range(len(y_true)):
      if y_true[i]==0.00:
        y_true[i]=eps
    return np.mean((np.abs(y_true - y_pred)) / np.abs(y_true)) * 100


x=pd.read_csv('/home/pn_kumar/Karthik/n50.csv',parse_dates=['Date'],index_col='Date')
x = x.loc["2016-01-01" :]                         #Since 2016-01-01, 5y(1234rows till 2020-12-31), + year 2021's rows (till 30th of April)
y=x.copy()                                        #deep copy
x.reset_index(drop=True, inplace=True)

stonks=[]
for i in x:
  stonks.append(i)
len(stonks)

alldata=x   #the original dataset

def create_dataset(dataset, time_step=1):         # convert an array of values into a dataset matrix which will be used to train the lstm model.
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step):
		a = dataset[i:(i+time_step), 0]               #i=0, 0,1,2,3-----(timesteps-1)  -> timesteps
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

after2020=len(y.loc["2021-01-01" : ])
timesteps=60
batchSize=32
dropout_value=0.15



def stacked_lstm_forecast(df1, timesteps, batchSize):
  scaler = MinMaxScaler(feature_range=(0, 1))
  df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))  # minmax scalar transformation of data

  before_2021_data_length = int(len(df1) - after2020)  #length of data before 2021
  training_size = int(before_2021_data_length * 0.80)  #80% of training size, refered from Yadav et al (2020) (Science Direct)
  train_data = df1[0:training_size, :]
  test_data = df1[training_size:before_2021_data_length,:1]  #20% of testing data, refered from Yadav et al (2020) (Science Direct)
  inpdata = df1[before_2021_data_length - timesteps:len(df1), :1]  #getting the data from 01-01-2021 onwards


  #reshape into X=t,t+1,t+2,t+3,........t+"timestep-1" and Y=t+"timestep"
  X_train, y_train = create_dataset(train_data, timesteps)
  x_inp, y_inp = create_dataset(inpdata, timesteps)
  x_test, y_test = create_dataset(test_data,timesteps)

  X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
  x_inp = x_inp.reshape(x_inp.shape[0],x_inp.shape[1] , 1)        #reshape input to be [samples, time steps, features] which is required for LSTM
  x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

  # initialising stacked lstm
  model=Sequential()
  model.add(LSTM(312, return_sequences=True, input_shape=(timesteps, 1), activation='tanh', dropout=0.1))
  model.add(LSTM(280, return_sequences=True, activation='tanh', dropout=0.1))
  model.add(LSTM(64, return_sequences=True, activation='tanh', dropout=0.1))
  model.add(LSTM(280, return_sequences=True, activation='tanh', dropout=0.1))
  model.add(LSTM(368, return_sequences=False, activation='tanh', dropout=0.1))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='mean_squared_error',optimizer='adam')

  model.fit(X_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=batchSize,verbose=1)     # training of the model

  test_predict=model.predict(x_test)                    #prediction using test data as input

  #performance metrics between, original test data and predicted test data
  msetst =mean_squared_error(y_test,test_predict)
  rmsetst=math.sqrt(msetst)
  maetst =mean_absolute_error(y_test,test_predict)
  r2tst  =r2_score(y_test,test_predict)
  mapetst=mean_absolute_percentage_error(y_test,test_predict)
  tstlst =[msetst,rmsetst,maetst,r2tst,mapetst]


  #model is trained again on the test data so as to increase the learning (it is often termed as incremental learning)
  #refered from url: https://www.justintodata.com/forecast-time-series-lstm-with-tensorflow-keras/#step-2-transforming-the-dataset-for-tensorflow-keras
  #refered from url: https://github.com/keras-team/keras/issues/4446
  model.fit(x_test,y_test,epochs=100,batch_size=batchSize,verbose=1)

  out_predict=model.predict(x_inp)                      #dynamic prediction of the stock's closing price from 01-01-2021 onwards

  #performance metrics between, original data(after 31-12-2020) and dynamically predicted data (after 31-12-2020)
  mseinp =mean_squared_error(y_inp,out_predict)
  rmseinp=math.sqrt(mseinp)
  maeinp =mean_absolute_error(y_inp,out_predict)
  r2inp  =r2_score(y_inp,out_predict)
  mapeinp=mean_absolute_percentage_error(y_inp,out_predict)
  inplst =[mseinp,rmseinp,maeinp,r2inp,mapeinp]



  lst=[]
  for i in out_predict:
    lst.append(i)

  p=train_data.tolist()
  q=test_data.tolist()
  p.extend(q)                                         #appending train and test data to make dataset before 2021 (data till 31-12-2020)
  p.extend(lst)                                       #appending the data, forcasted from 01-01-2021 onwards, to the data till 31-12-2020
  p=scaler.inverse_transform(p).tolist()

  return pd.DataFrame(p), tstlst, inplst
  #returns a dataframe, tstlst => test performance metrics, inplst => forcasted data performance metrics

mtest=[]
mdynamic=[]
fdata=pd.DataFrame()
for i in alldata:                                   # this for loop will be for each column of the original dataset
  temp=alldata[i]
  ftemp,trmse,drmse=stacked_lstm_forecast(temp, timesteps, batchSize)    #hyperparameters are provided as input here
  fdata = pd.concat([fdata,ftemp],axis = 1)
  mtest.append(trmse)
  mdynamic.append(drmse)
fdata.columns=stonks

fdata.to_csv('/home/pn_kumar/LSTM/fdata02_04_22.csv')
clm=['MSE','RMSE','MAE','R2','MAPE']
pd.DataFrame(mtest,index=stonks,columns=clm).to_csv('/home/pn_kumar/Karthik/06-02/mtest06_02_22.csv') #metric values saved
pd.DataFrame(mdynamic,index=stonks,columns=clm).to_csv('/home/pn_kumar/Karthik/06-02/mdynamic.csv') #metric values saved

