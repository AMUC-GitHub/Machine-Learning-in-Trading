from __future__ import division
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import pandas_datareader as web
import fbprophet
from datetime import datetime,date


def prophet(stock,base):
    # To Set up End and Start times for data grab
    start = datetime(2017,1,1)
    end = date.today()
    #To set DataFrame as the Stock Ticker
    df = web.DataReader(stock,'yahoo',start,end)['Close']
    
    df = df.reset_index()
    df[['ds','y']] = df[['Date' ,'Close']]
    df = df[['ds','y']]
    m = fbprophet.Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=56)
    forecast = m.predict(future)
    #print("forecast")
    #print(forecast.tail())
    forecast.drop(forecast.columns.difference(['ds','yhat']), 1, inplace=True)

    plt.figure(figsize = (12,6))
    close=df['y']
    pred = forecast['yhat']
    close.plot(label='MSFT previous value')
    
    pred.plot(label='Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    #plt.show()
    my_path = base + '/result/'
    filename='/prophet.jpg'
    plt.savefig(my_path+stock+filename)
