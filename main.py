import pandas as pd
import numpy as n
from moving_avg import moving_avg
from knn import knn
from prophet import prophet
from p_regression import polynomialRegression
from svm import supportVectorMachine
from l_regression import linearRegression
import xlrd
import os
 

base = os.path.abspath(__file__)
book = xlrd.open_workbook('YahooTickers.xlsx')
sheet = book.sheet_by_name('Stock')

data = [sheet.cell_value(r, 0) for r in range(sheet.nrows)]

ticker=data[5:10]
print(ticker)
for st in ticker:
	os.mkdir(os.path.join(base_path, st))
	moving_avg(st,base)
	knn(st,base)
	prophet(st,base)
	polynomialRegression(st,base)
	supportVectorMachine(st,base)
	linearRegression(st,base)
