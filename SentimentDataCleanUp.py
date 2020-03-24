
import pandas as pd

tweet_data = pd.read_csv("ProcessedData/SentimentAnalysisResults.csv")

# counts naive bayes sentiment results
naive_bayes_count = tweet_data.groupby(['CREATED AT'])['Naive Bayes'].value_counts()
naive_bayes_count = naive_bayes_count.to_frame()
naive_bayes = pd.DataFrame(naive_bayes_count)
naive_bayes.to_csv("ProcessedData/naive_bayes_count.csv")

# calculates mode of naive bayes results mode per day
naive_bayes_mode = tweet_data.groupby(['CREATED AT'])['Naive Bayes'].agg(pd.Series.mode)
naive_bayes_mode.to_csv("ProcessedData/naive_bayes_mode.csv")

# calculates median of naive bayes positive and negative tweets per day
naive_bayes_count_results = pd.read_csv("ProcessedData/naive_bayes_count.csv")
naive_bayes_median = naive_bayes_count_results.groupby(['CREATED AT'])['Count'].median()
naive_bayes_median = naive_bayes_median.to_frame()
naive_bayes_median = pd.DataFrame(naive_bayes_median)
naive_bayes_median.to_csv("ProcessedData/naive_bayes_median.csv")

# counts Multinomial Naive Bayes sentiment results
multinomial_naive_bayes_count = tweet_data.groupby(['CREATED AT'])['Multinomial Naive Bayes'].value_counts()
multinomial_naive_bayes_count = multinomial_naive_bayes_count.to_frame()
multinomial_naive_bayes_results = pd.DataFrame(multinomial_naive_bayes_count)
multinomial_naive_bayes_results.to_csv("ProcessedData/multinomial_naive_bayes_counts.csv")

# calculates mode of Multinomial naive bayes results mode per day
multinomial_naive_bayes_mode = tweet_data.groupby(['CREATED AT'])['Multinomial Naive Bayes'].agg(pd.Series.mode)
multinomial_naive_bayes_mode.to_csv("ProcessedData/multinomial_naive_bayes_mode.csv")

# calculates median of Multinomial naive bayes positive and negative tweets per day
multinomial_naive_bayes_count_results = pd.read_csv("ProcessedData/multinomial_naive_bayes_counts.csv")
multinomial_naive_bayes_median = multinomial_naive_bayes_count_results.groupby(['CREATED AT'])['Counts'].median()
multinomial_naive_bayes_median = multinomial_naive_bayes_median.to_frame()
multinomial_naive_bayes_median = pd.DataFrame(multinomial_naive_bayes_median)
multinomial_naive_bayes_median.to_csv("ProcessedData/multinomial_naive_bayes_median.csv")

# counts Bernoulli Naive Bayes sentiment results
bernoulli_naive_bayes_count = tweet_data.groupby(['CREATED AT'])['Bernoulli Naive Bayes'].value_counts()
bernoulli_naive_bayes_count = bernoulli_naive_bayes_count.to_frame()
bernoulli_naive_bayes_results = pd.DataFrame(bernoulli_naive_bayes_count)
bernoulli_naive_bayes_results.to_csv("ProcessedData/bernoulli_naive_bayes_counts.csv")

# calculates mode of Bernoulli naive bayes results mode per day
bernoulli_naive_bayes_mode = tweet_data.groupby(['CREATED AT'])['Bernoulli Naive Bayes'].agg(pd.Series.mode)
bernoulli_naive_bayes_mode.to_csv("ProcessedData/bernoulli_naive_bayes_mode.csv")

# calculates median of Bernoulli naive bayes positive and negative tweets per day
bernoulli_naive_bayes_count_results = pd.read_csv("ProcessedData/bernoulli_naive_bayes_counts.csv")
bernoulli_naive_bayes_median = bernoulli_naive_bayes_count_results.groupby(['CREATED AT'])['Counts'].median()
bernoulli_naive_bayes_median = bernoulli_naive_bayes_median.to_frame()
bernoulli_naive_bayes_median = pd.DataFrame(bernoulli_naive_bayes_median)
bernoulli_naive_bayes_median.to_csv("ProcessedData/bernoulli_naive_bayes_median.csv")

# counts Logistic Regression sentiment results
logistic_regression_count = tweet_data.groupby(['CREATED AT'])['Logistic Regression'].value_counts()
logistic_regression_count = logistic_regression_count.to_frame()
logistic_regression_results = pd.DataFrame(logistic_regression_count)
logistic_regression_results.to_csv("ProcessedData/logistic_regression_counts.csv")

# calculates mode of Logistic Regression results mode per day
logistic_regression_mode = tweet_data.groupby(['CREATED AT'])['Logistic Regression'].agg(pd.Series.mode)
logistic_regression_mode.to_csv("ProcessedData/logistic_regression_mode.csv")

# calculates median of Logistic Regression positive and negative tweets per day
logistic_regression_count_results = pd.read_csv("ProcessedData/logistic_regression_counts.csv")
logistic_regression_median = logistic_regression_count_results.groupby(['CREATED AT'])['Counts'].median()
logistic_regression_median = logistic_regression_median.to_frame()
logistic_regression_median = pd.DataFrame(logistic_regression_median)
logistic_regression_median.to_csv("ProcessedData/logistic_regression_median.csv")

# counts Linear Support Vector sentiment results
linear_support_vector_count = tweet_data.groupby(['CREATED AT'])['Linear Support Vector'].value_counts()
linear_support_vector_count = linear_support_vector_count.to_frame()
linear_support_vector_results = pd.DataFrame(linear_support_vector_count)
linear_support_vector_results.to_csv("ProcessedData/linear_support_vector_counts.csv")

# calculates mode of Linear Support Vector results mode per day
linear_support_vector_mode = tweet_data.groupby(['CREATED AT'])['Linear Support Vector'].agg(pd.Series.mode)
linear_support_vector_mode.to_csv("ProcessedData/linear_support_vector_mode.csv")

# calculates median of Linear Support Vector positive and negative tweets per day
linear_support_vector_count_results = pd.read_csv("ProcessedData/linear_support_vector_counts.csv")
linear_support_vector_median = linear_support_vector_count_results.groupby(['CREATED AT'])['Counts'].median()
linear_support_vector_median = linear_support_vector_median.to_frame()
linear_support_vector_median = pd.DataFrame(linear_support_vector_median)
linear_support_vector_median.to_csv("ProcessedData/linear_support_vector_median.csv")

# counts Nu-Support Vector sentiment results
nu_support_vector_count = tweet_data.groupby(['CREATED AT'])['Nu-Support Vector'].value_counts()
nu_support_vector_count = nu_support_vector_count.to_frame()
nu_support_vector_results = pd.DataFrame(nu_support_vector_count)
nu_support_vector_results.to_csv("ProcessedData/nu_support_vector_counts.csv")

# calculates mode of Nu-Support Vector results mode per day
nu_support_vector_mode = tweet_data.groupby(['CREATED AT'])['Nu-Support Vector'].agg(pd.Series.mode)
nu_support_vector_mode.to_csv("ProcessedData/nu_support_vector_mode.csv")

# calculates median of Nu-Support Vector positive and negative tweets per day
nu_support_vector_count_results = pd.read_csv("ProcessedData/nu_support_vector_counts.csv")
nu_support_vector_median = nu_support_vector_count_results.groupby(['CREATED AT'])['Counts'].median()
nu_support_vector_median = nu_support_vector_median.to_frame()
nu_support_vector_median = pd.DataFrame(nu_support_vector_median)
nu_support_vector_median.to_csv("ProcessedData/nu_support_vector_median.csv")

# counts Stochastic Gradient Descent sentiment results
stochastic_gradient_descent_count = tweet_data.groupby(['CREATED AT'])['Stochastic Gradient Descent'].value_counts()
stochastic_gradient_descent_count = stochastic_gradient_descent_count.to_frame()
stochastic_gradient_descent_result = pd.DataFrame(stochastic_gradient_descent_count)
stochastic_gradient_descent_result.to_csv("ProcessedData/stochastic_gradient_descent_counts.csv")

# calculates mode of Stochastic Gradient Descent results mode per day
stochastic_gradient_descent_mode = tweet_data.groupby(['CREATED AT'])['Stochastic Gradient Descent'].agg(pd.Series.mode)
stochastic_gradient_descent_mode.to_csv("ProcessedData/stochastic_gradient_descent_mode.csv")

# calculates median of Stochastic Gradient Descent positive and negative tweets per day
stochastic_gradient_descent_count_results = pd.read_csv("ProcessedData/stochastic_gradient_descent_counts.csv")
stochastic_gradient_descent_median = stochastic_gradient_descent_count_results.groupby(['CREATED AT'])[
    'Counts'].median()
stochastic_gradient_descent_median = stochastic_gradient_descent_median.to_frame()
stochastic_gradient_descent_median = pd.DataFrame(stochastic_gradient_descent_median)
stochastic_gradient_descent_median.to_csv("ProcessedData/stochastic_gradient_descent_median.csv")

