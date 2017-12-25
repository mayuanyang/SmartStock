import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

def get_stock_data(stock_name, normalized=0):
    url="data/0883.HK.csv"

    col_names = ['Date','Open','High','Low','Close','Adj close', 'Volume']
    stocks = pd.read_csv(url, header=0, names=col_names)[['Open','High','Volume','Close']]
    
    # stocks = pd.read_csv(url, index_col=False, usecols=cols_to_use)[cols_to_use]
    #print(stocks)
    df = pd.DataFrame(stocks)
    # df.drop(df.columns[[0,3,5]], axis=1, inplace=True) 
    return df

stock_name = '0883'
df = get_stock_data(stock_name,0)
df.tail()

today = datetime.date.today()
file_name = stock_name+'_stock_%s.csv' % today
print(file_name)
df.to_csv(file_name)

# df['High'] = df['High'] / 100
# df['Open'] = df['Open'] / 100
# df['Close'] = df['Close'] / 100
# df['Volume'] = df['Volume'] / 100
# print("The first 5 rows")
# print(df.head(5))

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    print('amount of features: ', amount_of_features)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]   
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    print("the training Y values")
    print(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='relu'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model

window = 5
X_train, y_train, X_test, y_test = load_data(df[::-1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

# model = build_model([3,lag,1])
numOfColumnsAsInout = 4
model = build_model2([numOfColumnsAsInout,window,1])
print(model.summary())

normalizationFactor = 10

model.fit(
    X_train/normalizationFactor,
    y_train/normalizationFactor,
    batch_size=64,
    nb_epoch=500,
    validation_split=0.1,
    verbose=1)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

# print(X_test[-1])
diff=[]
ratio=[]
p = model.predict(X_test)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))

import matplotlib.pyplot as plt2

plt2.plot(p,color='red', label='prediction')
plt2.plot(y_test,color='blue', label='y_test')
plt2.legend(loc='upper left')
plt2.show()