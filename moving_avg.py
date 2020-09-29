import keras
import os
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd 
import pandas_datareader.data as web
import datetime
from datetime import date
import numpy as np 
from matplotlib import style
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def moving_avg(st,base):
	# Get the stock data using yahoo API:
	style.use('ggplot')

	# get data to train our model
	start = datetime.datetime(2017,9,12)
	end = date.today()
	df = web.DataReader(st, 'yahoo', start, end) 

	# fix the date 
	df.reset_index(inplace=True)
	df.set_index("Date", inplace=True)

	df.tail()
	# Rolling mean
	close_px = df['Close']
	mavg10 = close_px.rolling(window=10).mean()
	mavg30 = close_px.rolling(window=30).mean()
	mavg60 = close_px.rolling(window=60).mean()
	plt.figure(figsize = (12,6))
	close_px.plot(label=st)
	mavg10.plot(label='moving average 10 days')
	mavg30.plot(label='moving average 30 days')
	mavg60.plot(label='moving average 60 days')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()
	#plt.show()
	#folder=os.path.join(base,st)
	my_path = base + '/result/'
	filename='/moving_avg.jpg'
	plt.savefig(my_path+st+filename)
