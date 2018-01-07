# SmartStock
Use Keras, Tensorflow and LSTM to predict stock price

## How to start
Make sure you have Python, Tensorflow and Keras installed
- Tensorflow installation guide https://www.tensorflow.org/install/
- Keras documentation https://keras.io/

Once the environment is ready, just simply open a command prompt and type in `python multiinputs.py`

## How it works
I treat it as a supervised training problem, the csv file contains multiple columns e.g. `['Open','High','Low','Close','Adj close', 'Volume']`,
the process will preprocess the data by appending the next day's columns to the right as the `answer` and use `LSTM` to work it out.

The data will be split to 90%/10% for training and evaluation, here is the example result

![image](https://user-images.githubusercontent.com/3387099/34645411-ab279e96-f398-11e7-861f-82656bf73eb2.png)
