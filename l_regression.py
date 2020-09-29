import pandas as pd
import numpy as np
import datetime
from datetime import date
import pandas_datareader.data as web
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def linearRegression(st,base):
	#Extract the data file
	start = datetime.datetime(2017,1,1)
	end = date.today()
	df = web.DataReader(st, 'yahoo', start, end) 

	#setting index as date
	df = df.sort_values('Date')
	df.reset_index(inplace=True)
	df.set_index("Date", inplace=True)

	#converting dates into number of days as dates cannot be passed directly to any regression model
	df.index = (df.index - pd.to_datetime('2000-9-12')).days

	# Convert the pandas series into numpy array, we need to further massage it before sending it to regression model
	y = np.asarray(df['Close'])
	x = np.asarray(df.index.values)

	# Model initialization
	regression_model = LinearRegression()

	# Fit the data(train the model)
	regression_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

	# Prediction for historical dates. Let's call it learned values.
	y_learned = regression_model.predict(x.reshape(-1, 1))

	newindex = np.asarray(pd.RangeIndex(start=x[-1], stop=x[-1] + 56))

	# Prediction for future dates. Let's call it predicted values.
	y_predict = regression_model.predict(newindex.reshape(-1, 1))

	#print the last predicted value
	print ("Closing price at 2 months would be around ", y_predict[-1])

	#convert the days index back to dates index for plotting the graph
	x = pd.to_datetime(df.index, origin='2000-9-12', unit='D')
	future_x = pd.to_datetime(newindex, origin='2000-9-12', unit='D')

	#setting figure size
	from matplotlib.pylab import rcParams
	rcParams['figure.figsize'] = 20,10

	#plot the actual data
	plt.figure(figsize=(16,8))
	plt.plot(x,df['Close'], label='Close Price History from 2017')

	#plot the regression model
	plt.plot(x,y_learned, color='r', label='Mathematical Model')

	#plot the future predictions
	plt.plot(future_x,y_predict, color='g', label='Future predictions')

	plt.suptitle('Stock Market Predictions', fontsize=16)

	fig = plt.gcf()
	fig.canvas.set_window_title('Predictions')
	plt.legend()
	#plt.show()
	my_path = base + '/result/'
	filename='/linearRegression.jpg'
	plt.savefig(my_path+st+filename)
