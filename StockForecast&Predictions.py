# this module predicts january 2020 prices based on the historical data from 2019
# prediction is done using ML algo and Adj Close 2019 prices
import time
start_counting =time.time()
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# read data into dataframe, set index as dates and specify column from which values will be taken for further
# forecast and further ML algo predicitons
stock_data = pd.read_csv("OutputData/2019-20_BTC-USD.csv")
stock_data = stock_data.set_index('Date')
stock_data = pd.read_csv("OutputData/2019-20_BTC-USD.csv")

# daily volatility difference,
stock_data['HL_%'] = (stock_data['High'] - stock_data['Low']) / stock_data['Adj Close']
# daily percentage change
stock_data['%_change'] = (stock_data['Adj Close'] - stock_data['Open']) / stock_data['Open']

stock_data = stock_data[['Adj Close', 'HL_%', '%_change', 'Volume', 'Date']]
stock_data = stock_data.set_index('Date')

print(stock_data)

# A variable for predicting 'n' days out into the future
forecast_out = 32  # 'n=32' days (all January 2019 + 1st of Feb)
# Create another column (the target ) shifted 'n' units up
stock_data['Prediction'] = stock_data['Adj Close'].shift(-forecast_out)
# print the new data set
print(stock_data)

# Convert the dataframe to a numpy array
X = np.array(stock_data.drop(['Prediction'], 1))

# Remove the last '31' rows (all the January 2019 values)
X = X[:-forecast_out]

# dataframe to a numpy array
y = np.array(stock_data['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]

# Setting 80% training and 20% testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Create and train the Support Vector Regression
svr_rbf = SVR()
svr_rbf.fit(x_train, y_train)

# Testing score  of svr and value of the confidence
svr_confidence = svr_rbf.score(x_test, y_test)
print("Support Vector Regression confidence: ", (svr_confidence) * 100, "%")

# Create and train the Linear Regression  Model
lr = LinearRegression(n_jobs=-1)
# Train the model
lr.fit(x_train, y_train)

# Testing score of lr and value of the confidence
lr_confidence = lr.score(x_test, y_test)
print("Linear Regression confidence: ", (lr_confidence) * 100, "%", "\n")

# Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(stock_data.drop(['Prediction'], 1))[-forecast_out:]
print("January 2019 forecast: ", "\n", x_forecast, "\n")

# Print linear regression model predictions for the next '30' days and converts array to dataframe
lr_prediction = lr.predict(x_forecast)
print("Liner Regression January prediction:", "\n", lr_prediction, "\n")
lr_prediction_series = pd.Series(lr_prediction)
lr_prediction_frame = lr_prediction_series.to_frame("LR Prediction")

# Print support vector regressor model predictions for the next '30' days and converts array to dataframe
svr_prediction = svr_rbf.predict(x_forecast)
print("Support Vector Regression January 2019 predictions:", "\n", svr_prediction)
svr_prediction_series = pd.Series(svr_prediction)
svr_prediction_frame = svr_prediction_series.to_frame("SVR Prediction")

# dropping dataframe index to be able to link them all together
stock_data.reset_index(drop=False, inplace=True)
lr_prediction_frame.reset_index(drop=True, inplace=True)
svr_prediction_frame.reset_index(drop=True, inplace=True)

# linking all 3 dataframes into one and re-setting the index to dates, save to csv
stock_data_predictions = pd.concat([stock_data, lr_prediction_frame, svr_prediction_frame], axis=1)
stock_data_predictions = stock_data_predictions.set_index('Date', drop=True)
stock_data_predictions.to_csv("ProcessedData/StockPredictions.csv")

