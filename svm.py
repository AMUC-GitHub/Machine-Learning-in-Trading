from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas_datareader.data as web
import datetime as dt
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def supportVectorMachine(st,base):
	df=web.DataReader(st,'yahoo',start='2017-01-01',end=date.today())
	df=df[['Close']]
	df['Prediction'] = df[['Close']].shift(-56)
	#Create a data set X and convert it into numpy array , which will be having actual values
	X = np.array(df.drop(['Prediction'],1))
	#Remove the last 56 rows
	X = X[:-56]
	# Create a dataset y which will be having Predicted values and convert into numpy array
	y = np.array(df['Prediction'])
	# Remove Last 56 rows
	y = y[:-56]
	# Split the data into train and test with 90 & 10 % respectively
	#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	svr = SVR(kernel="rbf", C=1e3, gamma=0.1)
	svr.fit(X, y)
	forecast = np.array(df.drop(['Prediction'],1))[-56:]
	#print("forecast")
	#print(df.__dict__)

	# support vector model predictions 
	svm_prediction = svr.predict(forecast)

	plt.figure(figsize = (12,6))
	df_new = pd.DataFrame(y)
	df_new.columns = ['y']

	# Retrieve index values
	new_index = df_new['y'].tail(len(svm_prediction)).index

	# Make a dataframe with your prediction values and your index
	new_series = pd.DataFrame(index = new_index, data = svm_prediction)

	# Merge the dataframes
	df_new = pd.merge(df_new, new_series, how = 'left', left_index=True, right_index=True)
	df_new.columns = ['y', 'svm_prediction']


	plt.plot(df_new['y'][:-56], color="r", label=' Data from 2017')
	plt.plot(df_new['svm_prediction'], color="b", label='Predicted Data')

	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()
	#plt.show()
	my_path = base + '/result/'
	filename='/svm.jpg'
	plt.savefig(my_path+st+filename)
