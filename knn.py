import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as web
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def knn(st,base):
	df=web.DataReader(st,data_source='yahoo',start='2017-01-01',end=date.today())
	df.reset_index(inplace=True,drop=False)
	df = df.sort_values('Date')

	df['Year']=df['Date'].dt.year
	df['Month']=df['Date'].dt.month
	df['Day']=df['Date'].dt.day

	df=df[['Day','Month','Year','High','Open','Low','Close']]

	#separate Independent and dependent variable
	X = df.iloc[:,df.columns !='Close']
	Y= df['Close']

	x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=.17)

	knn_regressor=KNeighborsRegressor(n_neighbors = 5)
	knn_model=knn_regressor.fit(x_train,y_train)
	y_learned = knn_model.predict(x_train)
	y_knn_prediction=knn_model.predict(x_test)

	plt.figure(figsize=(16,8))
	plt.title('Stocks Closing Price')
	y=df['Close']
	df_new = pd.DataFrame(y)
	df_new.columns = ['y']

	# Retrieve index values
	new_index = df_new['y'].tail(len(y_knn_prediction)).index

	# Make a dataframe with your prediction values and your index
	new_series = pd.DataFrame(index = new_index, data = y_knn_prediction)

	# Merge the dataframes
	df_new = pd.merge(df_new, new_series, how = 'left', left_index=True, right_index=True)
	df_new.columns = ['y', 'y_knn_prediction']

	plt.plot(df_new['y'][:-56], color="r", label='Data from 2017')
	plt.plot(df_new['y_knn_prediction'], color="g", label='Predicted Data')
	#plt.plot(y_learned)
	plt.xlabel('Date',fontsize=18)
	plt.ylabel('Close Price US($)',fontsize=18)
	plt.style.use('fivethirtyeight')
	plt.legend()
	#plt.show()
	my_path = base + '/result/'
	filename='/knn.jpg'
	plt.savefig(my_path+st+filename)

