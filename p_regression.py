import numpy as np
import datetime
from datetime import date
import pandas_datareader.data as web
import pandas as pd 
from sklearn.linear_model import LinearRegression
from matplotlib.pylab import rcParams
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def polynomialRegression(st,base):
	#Extract the data file
	start = datetime.datetime(2017,1,1)
	end = date.today()
	df = web.DataReader(st, 'yahoo', start, end) 

	#Setting index as date
	df = df.sort_values('Date')
	df.reset_index(inplace=True)
	df.set_index("Date", inplace=True)
	#print(df.__dict__)

	#Converting dates into number of days as dates cannot be passed directly 
	#to any regression model
	df.index = (df.index - pd.to_datetime('2000-9-12')).days

	#Convert the pandas series into numpy array, we need to further 
	#massage it before sending it to regression model
	y = np.asarray(df['Close'])
	x = np.asarray(df.index.values)

	#Model initialization
	regression_model = LinearRegression()

	#Choose the order of your polynomial. Here the degree is set to 5.
	#hence the mathematical model equation is 
	#y = c0 + c1.x**1 + c2.x**2+....+ c5.x**5
	poly = PolynomialFeatures(5)

	#Convert dimension x in the higher degree polynomial expression
	X_transform = poly.fit_transform(x.reshape(-1, 1))

	#Fit the data(train the model)
	regression_model.fit(X_transform, y.reshape(-1, 1))

	# Prediction for historical dates. Let's call it learned values.
	y_learned = regression_model.predict(X_transform)

	#Now, add future dates to the date index and pass that index to 
	#the regression model for future prediction.
	newindex = np.asarray(pd.RangeIndex(start=x[-1], stop=x[-1] + 56))

	#Convert the extended dimension x in the higher degree polynomial expression
	X_extended_transform = poly.fit_transform(newindex.reshape(-1, 1))

	y_predict = regression_model.predict(X_extended_transform)

	#last predicted value
	print("2 months ",y_predict[-1])

	x=pd.to_datetime(df.index, origin='2000-9-12', unit='D')
	future_x=pd.to_datetime(newindex, origin='2000-9-12', unit='D')

	#figure size

	rcParams['figure.figsize'] = 20,10
	plt.figure(figsize = (16,8))
	plt.plot(x, df['Close'], color= 'black', label= 'Close Price History from 2017')
	plt.plot(x, y_learned, color= 'red', label= 'Polynomial model')
	plt.plot(future_x,y_predict, color='g', label='Future Predictions') 
	#plt.xlabel('Date')
	#plt.ylabel('Price')
	fig=plt.gcf()
	fig.canvas.set_window_title('prediction')
	plt.legend()
	#display the graph
	#plt.show()
	my_path = base + '/result/'
	filename='/polynomialRegression.jpg'
	plt.savefig(my_path+st+filename)
