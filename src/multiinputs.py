from math import sqrt

import datetime
from keras.callbacks import ReduceLROnPlateau
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import math

def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

# convert series to supervised learning, it appends the predict columns to the right of the dataset, so if original dataset has 7 columns, it will become 14
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    print(df.head())
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-(i))) # Can shift the close price to be 2 days later (i + 1), we train the network to predict this price
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    print("shifted")
    print(agg.head())
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
url="data/0883.HK_weekly.csv"
col_names = ['Open','High','Low','Close','Adj close', 'Volume']
dataset = pd.read_csv(url, header=0, names=col_names)[['Open','Low','High','Volume','Adj close','Close']]
dataset.columns = ['Open','Low','High','Volume','Adj close','Close']

dataset.to_csv("test.csv")

values = dataset.values
for i in range(0, len(values) - 1):
    for j in range(0, len(values[i])):
        #print(vv)
        if (math.isnan(values[i][j])):
            values[i][j] = 0


#print values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[6,7,8,9,10]], axis=1, inplace=True) # Drop PrevClose in prediction
print(reframed.head())

# split into train and test sets
values = reframed.values
row = int(round(0.9 * reframed.shape[0]))
train = values[:row, :]
test = values[row:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# callbacks
lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=50, cooldown=0, verbose=1)

# design network
dropRate = 0.2
model = Sequential()
#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=500, batch_size=1, verbose=1,
                    shuffle=False, validation_data=(test_X, test_y), callbacks=[lr_reducer])
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))


print("Predict")
print(inv_yhat)
print("Actual")
print(inv_y) # The test set was used
print('Test RMSE: %.3f' % rmse)

pyplot.plot(inv_yhat, label='Predict')
pyplot.plot(inv_y, label='Actual')
pyplot.legend()
pyplot.show()