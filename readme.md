# Stock Price Prediction

Prediction of future stock prices is done using different algorithms of Machine Learning. The data is extracted from Yahoo API using [pandas_datareader](https://pandas-datareader.readthedocs.io/en/latest/) package.

## Description

The Algorithms used are :-
1. Moving Average 
2. Support Vector Machine
3. Linear Regression
4. Polynomial Regression
5. Prophet
6. K Nearest Neighbor

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the depencies.

```bash
pip3 install pandas
pip3 install numpy
pip3 install sklearn

```

## Implementation
For first 5 companies from the table in YahooTickers.xlsx file, the result is in the result folder.
To run the code follow the below instructions :

```python
cd Stock_Price_Prediction

python3 main.py
```
To get the result for all the companies :

```python
ticker=data[5:10]     #change this line
ticker=data[5:]       # and write this line
```
Individual algorithm can be used by passing the ticker of the company and path of the project directory

``` python
import knn from knn	#for K nearest neighbor algorithm
knn('MSFT',base)        #MSFT is the ticker for Microsoft and base is the path of directory
```

## Project Structure
Inside the Stock_Price_Prediction directory, main.py file runs all the functions of the algorithm which are in different files inside Stock_Price_Prediction folder. The result folder contains folder for each company which contains result graph obtained from each algorithm.
