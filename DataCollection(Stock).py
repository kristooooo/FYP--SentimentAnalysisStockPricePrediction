# packages required to download cryptocurrency data from web

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web

# time frame of the crypto market stock data required
start = dt.datetime(2019, 1, 1)
end = dt.datetime(2020, 1, 31)

# specify index code , source and time frame
df = web.DataReader('BTC-USD', 'yahoo', start, end)

# save output
df.to_csv('OutputData/2019-20_BTC-USD.csv')
df = pd.read_csv('OutputData/2019-20_BTC-USD.csv', parse_dates=True, index_col=0)

# adjusting close prices as these will be used to draw the trend
df['Adj Close'].plot(figsize=(15, 5))

# display the outcome
plt.show()
