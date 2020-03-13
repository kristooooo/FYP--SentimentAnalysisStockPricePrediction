import pandas as pd

######## calculations for Naive Bayes price predictions ##############

stock_data_january2020 = pd.read_csv("OutputData/2020Jan_BTC-USD.csv", usecols=['Adj Close'])
naive_bayes_mode = pd.read_csv("ProcessedData/naive_bayes_mode.csv", usecols=['Range', 'CREATED AT'])
naive_bayes_mode.set_index('CREATED AT')

# dropping dataframe index to be able to link them all together
stock_data_january2020.reset_index(drop=True, inplace=True)
naive_bayes_mode.reset_index(drop=True, inplace=True)

# linking all 2 dataframes into one and re-setting the index to dates, save to csv
naive_bayes_forecast = pd.concat([naive_bayes_mode, stock_data_january2020], axis=1)

naive_bayes_prices = []

# loop and calculate prices based on the amount of positive/negative tweets per day
for i in naive_bayes_forecast.itertuples():
    naive_bayes_range = int(i[2])
    price_nb = int(i[3])
    naive_bayes_price = (naive_bayes_range * 100 / price_nb) + price_nb
    naive_bayes_prices.append(naive_bayes_price)

naive_bayes_forecast['Naive Bayes prices'] = naive_bayes_prices
print(naive_bayes_forecast, "\n")
naive_bayes_forecast.to_csv("ProcessedData/naive_bayes_forecast.csv")

######## calculations for Nu-Support Vector price predictions ##############

nu_support_vector_mode = pd.read_csv("ProcessedData/nu_support_vector_mode.csv", usecols=['Range', 'CREATED AT'])
nu_support_vector_mode.set_index('CREATED AT')

# dropping dataframe index to be able to link them all together
stock_data_january2020.reset_index(drop=True, inplace=True)
nu_support_vector_mode.reset_index(drop=True, inplace=True)

# linking all 2 dataframes into one and re-setting the index to dates, save to csv
nu_support_vector_forecast = pd.concat([nu_support_vector_mode, stock_data_january2020], axis=1)

nu_support_vector_prices = []

# loop and calculate prices based on the amount of positive/negative tweets per day
for i in nu_support_vector_forecast.itertuples():
    nu_support_vector_range = int(i[2])
    price_nsv = int(i[3])
    nu_support_vector_price = (nu_support_vector_range * 100 / price_nsv) + price_nsv
    nu_support_vector_prices.append(nu_support_vector_price)

nu_support_vector_forecast['Nu-Support Vector prices'] = nu_support_vector_prices
print(nu_support_vector_forecast, "\n")
nu_support_vector_forecast.to_csv("ProcessedData/nu_support_vector_forecast.csv")

######## calculations for Stochastic Gradient Descent price predictions ##############

stochastic_gradient_descent_mode = pd.read_csv("ProcessedData/stochastic_gradient_descent_mode.csv",
                                               usecols=['Range', 'CREATED AT'])
stochastic_gradient_descent_mode.set_index('CREATED AT')

# dropping dataframe index to be able to link them all together
stock_data_january2020.reset_index(drop=True, inplace=True)
stochastic_gradient_descent_mode.reset_index(drop=True, inplace=True)

# linking all 2 dataframes into one and re-setting the index to dates, save to csv
stochastic_gradient_descent_forecast = pd.concat([stochastic_gradient_descent_mode, stock_data_january2020], axis=1)

stochastic_gradient_descent_prices = []

# loop and calculate prices based on the amount of positive/negative tweets per day
for i in stochastic_gradient_descent_forecast.itertuples():
    stochastic_gradient_descent_range = int(i[2])
    price_sgd = int(i[3])
    stochastic_gradient_descent_price = (stochastic_gradient_descent_range * 100 / price_sgd) + price_sgd
    stochastic_gradient_descent_prices.append(stochastic_gradient_descent_price)

stochastic_gradient_descent_forecast['Stochastic Gradient Descent prices'] = stochastic_gradient_descent_prices
print(stochastic_gradient_descent_forecast, "\n")
stochastic_gradient_descent_forecast.to_csv("ProcessedData/stochastic_gradient_descent_forecast.csv")

######## calculations for  Multinomial Naive Bayes price predictions ##############

multinomial_naive_bayes_mode = pd.read_csv("ProcessedData/multinomial_naive_bayes_mode.csv",
                                           usecols=['Range', 'CREATED AT'])
multinomial_naive_bayes_mode.set_index('CREATED AT')

# dropping dataframe index to be able to link them all together
stock_data_january2020.reset_index(drop=True, inplace=True)
multinomial_naive_bayes_mode.reset_index(drop=True, inplace=True)

# linking all 2 dataframes into one and re-setting the index to dates, save to csv
multinomial_naive_bayes_forecast = pd.concat([multinomial_naive_bayes_mode, stock_data_january2020], axis=1)

multinomial_naive_bayes_prices = []

# loop and calculate prices based on the amount of positive/negative tweets per day
for i in multinomial_naive_bayes_forecast.itertuples():
    multinomial_naive_bayes_range = int(i[2])
    price_mnb = int(i[3])
    multinomial_naive_bayes_price = (multinomial_naive_bayes_range * 100 / price_mnb) + price_mnb
    multinomial_naive_bayes_prices.append(multinomial_naive_bayes_price)

multinomial_naive_bayes_forecast['Multinomial Naive Bayes prices'] = multinomial_naive_bayes_prices
print(multinomial_naive_bayes_forecast, "\n")
multinomial_naive_bayes_forecast.to_csv("ProcessedData/multinomial_naive_bayes_forecast.csv")

######## calculations for Logistic Regression price predictions ##############

logistic_regression_mode = pd.read_csv("ProcessedData/logistic_regression_mode.csv", usecols=['Range', 'CREATED AT'])
logistic_regression_mode.set_index('CREATED AT')

# dropping dataframe index to be able to link them all together
stock_data_january2020.reset_index(drop=True, inplace=True)
logistic_regression_mode.reset_index(drop=True, inplace=True)

# linking all 2 dataframes into one and re-setting the index to dates, save to csv
logistic_regression_forecast = pd.concat([logistic_regression_mode, stock_data_january2020], axis=1)

logistic_regression_prices = []

# loop and calculate prices based on the amount of positive/negative tweets per day
for i in multinomial_naive_bayes_forecast.itertuples():
    logistic_regression_range = int(i[2])
    price_lr = int(i[3])
    logistic_regression_price = (logistic_regression_range * 100 / price_lr) + price_lr
    logistic_regression_prices.append(logistic_regression_price)

logistic_regression_forecast['Logistic Regression prices'] = logistic_regression_prices
print(logistic_regression_forecast, "\n")
logistic_regression_forecast.to_csv("ProcessedData/logistic_regression_forecast.csv")

######## calculations for Linear Support Vector price predictions ##############

linear_support_vector_mode = pd.read_csv("ProcessedData/linear_support_vector_mode.csv",
                                         usecols=['Range', 'CREATED AT'])
linear_support_vector_mode.set_index('CREATED AT')

# dropping dataframe index to be able to link them all together
stock_data_january2020.reset_index(drop=True, inplace=True)
linear_support_vector_mode.reset_index(drop=True, inplace=True)

# linking all 2 dataframes into one and re-setting the index to dates, save to csv
linear_support_vector_forecast = pd.concat([linear_support_vector_mode, stock_data_january2020], axis=1)

linear_support_vector_prices = []

# loop and calculate prices based on the amount of positive/negative tweets per day
for i in linear_support_vector_forecast.itertuples():
    linear_support_vector_range = int(i[2])
    price_lsv = int(i[3])
    linear_support_vector_price = (linear_support_vector_range * 100 / price_lsv) + price_lsv
    linear_support_vector_prices.append(linear_support_vector_price)

linear_support_vector_forecast['Linear Support Vector prices'] = linear_support_vector_prices
print(linear_support_vector_forecast, "\n")
linear_support_vector_forecast.to_csv("ProcessedData/linear_support_vector_forecast.csv")

######## calculations for Bernoulli Naive Bayes price predictions ##############

bernoulli_naive_bayes_mode = pd.read_csv("ProcessedData/bernoulli_naive_bayes_mode.csv",
                                         usecols=['Range', 'CREATED AT'])
bernoulli_naive_bayes_mode.set_index('CREATED AT')

# dropping dataframe index to be able to link them all together
stock_data_january2020.reset_index(drop=True, inplace=True)
bernoulli_naive_bayes_mode.reset_index(drop=True, inplace=True)

# linking all 2 dataframes into one and re-setting the index to dates, save to csv
bernoulli_naive_bayes_forecast = pd.concat([bernoulli_naive_bayes_mode, stock_data_january2020], axis=1)

bernoulli_naive_bayes_prices = []

# loop and calculate prices based on the amount of positive/negative tweets per day
for i in bernoulli_naive_bayes_forecast.itertuples():
    bernoulli_naive_byes_range = int(i[2])
    price_bnb = int(i[3])
    bernoulli_naive_byes_price = (bernoulli_naive_byes_range * 100 / price_bnb) + price_bnb
    bernoulli_naive_bayes_prices.append(bernoulli_naive_byes_price)

bernoulli_naive_bayes_forecast['Bernoulli Naive Bayes prices'] = bernoulli_naive_bayes_prices
print(bernoulli_naive_bayes_forecast, "\n")
bernoulli_naive_bayes_forecast.to_csv("ProcessedData/bernoulli_naive_bayes_forecast.csv")
