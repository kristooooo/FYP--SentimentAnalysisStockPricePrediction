import pandas as pd

tweet_data = pd.read_csv("ProcessedData/SentimentAnalysisResults.csv")

naive_bayes_count = tweet_data.groupby(['CREATED AT'])['Naive Bayes'].value_counts()
naive_bayes_count = naive_bayes_count.to_frame()
naive_bayes = pd.DataFrame(naive_bayes_count)
naive_bayes.to_csv("ProcessedData/naive_bayes_count.csv")
naive_bayes_mode = tweet_data.groupby(['CREATED AT'])['Naive Bayes'].agg(pd.Series.mode)
naive_bayes_mode.to_csv("ProcessedData/naive_bayes_mode.csv")

multinomial_naive_bayes_count = tweet_data.groupby(['CREATED AT'])['Multinomial Naive Bayes'].value_counts()
multinomial_naive_bayes_count = multinomial_naive_bayes_count.to_frame()
multinomial_naive_bayes_results = pd.DataFrame(multinomial_naive_bayes_count)
multinomial_naive_bayes_results.to_csv("ProcessedData/multinomial_naive_bayes_counts.csv")
multinomial_naive_bayes_mode = tweet_data.groupby(['CREATED AT'])['Multinomial Naive Bayes'].agg(pd.Series.mode)
multinomial_naive_bayes_mode.to_csv("ProcessedData/multinomial_naive_bayes_mode.csv")

bernoulli_naive_bayes_count = tweet_data.groupby(['CREATED AT'])['Bernoulli Naive Bayes'].value_counts()
bernoulli_naive_bayes_count = bernoulli_naive_bayes_count.to_frame()
bernoulli_naive_bayes_results = pd.DataFrame(bernoulli_naive_bayes_count)
bernoulli_naive_bayes_results.to_csv("ProcessedData/bernoulli_naive_bayes_counts.csv")
bernoulli_naive_bayes_mode = tweet_data.groupby(['CREATED AT'])['Bernoulli Naive Bayes'].agg(pd.Series.mode)
bernoulli_naive_bayes_mode.to_csv("ProcessedData/bernoulli_naive_bayes_mode.csv")

logistic_regression_count = tweet_data.groupby(['CREATED AT'])['Logistic Regression'].value_counts()
logistic_regression_count = logistic_regression_count.to_frame()
logistic_regression_results = pd.DataFrame(logistic_regression_count)
logistic_regression_results.to_csv("ProcessedData/logistic_regression_counts.csv")
logistic_regression_mode = tweet_data.groupby(['CREATED AT'])['Logistic Regression'].agg(pd.Series.mode)
logistic_regression_mode.to_csv("ProcessedData/logistic_regression_mode.csv")

linear_support_vector_count = tweet_data.groupby(['CREATED AT'])['Linear Support Vector'].value_counts()
linear_support_vector_count = linear_support_vector_count.to_frame()
linear_support_vector_results = pd.DataFrame(linear_support_vector_count)
linear_support_vector_results.to_csv("ProcessedData/linear_support_vector_counts.csv")
linear_support_vector_mode = tweet_data.groupby(['CREATED AT'])['Linear Support Vector'].agg(pd.Series.mode)
linear_support_vector_mode.to_csv("ProcessedData/linear_support_vector_mode.csv")

nu_support_vector_count = tweet_data.groupby(['CREATED AT'])['Nu-Support Vector'].value_counts()
nu_support_vector_count = nu_support_vector_count.to_frame()
nu_support_vector_results = pd.DataFrame(nu_support_vector_count)
nu_support_vector_results.to_csv("ProcessedData/nu_support_vector_counts.csv")
nu_support_vector_mode = tweet_data.groupby(['CREATED AT'])['Nu-Support Vector'].agg(pd.Series.mode)
nu_support_vector_mode.to_csv("ProcessedData/nu_support_vector_mode.csv")

stochastic_gradient_descent_count = tweet_data.groupby(['CREATED AT'])['Stochastic Gradient Descent'].value_counts()
stochastic_gradient_descent_count = stochastic_gradient_descent_count.to_frame()
stochastic_gradient_descent_result = pd.DataFrame(stochastic_gradient_descent_count)
stochastic_gradient_descent_result.to_csv("ProcessedData/stochastic_gradient_descent_counts.csv")
stochastic_gradient_descent_mode = tweet_data.groupby(['CREATED AT'])['Stochastic Gradient Descent'].agg(pd.Series.mode)
stochastic_gradient_descent_mode.to_csv("ProcessedData/stochastic_gradient_descent_mode.csv")

