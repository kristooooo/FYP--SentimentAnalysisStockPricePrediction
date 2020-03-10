import pickle
import re
import string

import chardet
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


# similarly to sample data method, remove_noise method are cleaning the sentences from unwanted features and
# converts a sentence into a list of words
def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub(r'<.*?>', '', token)  # remove HTML tags.

        token = re.sub(r'[^\x00-\x7F]+', '', token)
        token = re.sub(r'[^\w\s]', '', token)  # remove punc.
        token = re.sub(r'\d+', '', token)  # remove numbers
        token = token.lower()  # lower case, .upper() for upper
        token = re.sub(r'https*', "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# from here we are loading our trained algorithms so we can use them straight away, which will saves lots of time
open_file = open("TrainedAlgorithms/NB_classifier.pickle", "rb")
NB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("TrainedAlgorithms/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("TrainedAlgorithms/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("TrainedAlgorithms/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("TrainedAlgorithms/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("TrainedAlgorithms/NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("TrainedAlgorithms/SGDC_classifier.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()


# this is a method whoch recognise the encoding of the file we have generated when downloading our Tweets
def find_encoding(fname):
    r_file = open(fname, 'rb').read()
    encode_detect = chardet.detect(r_file)
    charenc = encode_detect['encoding']
    return charenc


# load our collected Tweets csv file into a dataframe and convert Tweet context column into a string
my_encoding = find_encoding("OutputData/CryptoTweetDataNoRetweetsID.csv")
tweet_data = pd.read_csv("OutputData/CryptoTweetDataNoRetweetsID.csv", encoding=my_encoding)
tweet_data['TEXT'] = tweet_data['TEXT'].apply(str)

# creating list for every algorithm where results of sentiment analysis will be inserted before we implement each
# list into a individual column in our dataframe
NB_results = []
MNB_results = []
BNB_results = []
LG_results = []
LSVC_results = []
NuSVC_results = []
SGDC_results = []

# loop for number of index in our dataframe
# gets the content of the tweet from TEXT column, removes the noise and converts string into words list
for column in tweet_data.itertuples():
    tweet = (getattr(column, 'TEXT'))
    clean_tweet = remove_noise(word_tokenize(tweet))

    # from here every single algorithm is taking cleaned tweets and provides a result if the tweet is negative or
    # positive and inerts the results into list accordingly
    NB_classifier_result = NB_classifier.classify(dict([token, True] for token in clean_tweet))
    print("Naive Bayes:", NB_classifier_result)
    NB_results.append(NB_classifier_result)

    MNB_classifier_result = MNB_classifier.classify(dict([token, True] for token in clean_tweet))
    print("Multinomial Naive Bayes:", MNB_classifier_result)
    tweet_data['Multinomial Naive Bayes'] = MNB_classifier_result
    MNB_results.append(MNB_classifier_result)

    BernoulliNB_classifier_result = BernoulliNB_classifier.classify(dict([token, True] for token in clean_tweet))
    print("Bernoulli Naive Bayes:", BernoulliNB_classifier_result)
    tweet_data['Bernoulli Naive Bayes'] = BernoulliNB_classifier_result
    BNB_results.append(BernoulliNB_classifier_result)

    LogisticRegression_classifier_result = LogisticRegression_classifier.classify(
        dict([token, True] for token in clean_tweet))
    print("Logistic Regression:", LogisticRegression_classifier_result)
    tweet_data['Logistic Regression'] = LogisticRegression_classifier_result
    LG_results.append(LogisticRegression_classifier_result)

    LinearSVC_classifier_result = LinearSVC_classifier.classify(dict([token, True] for token in clean_tweet))
    print("Linear Support Vector:", LinearSVC_classifier_result)
    tweet_data['Linear Support Vector'] = LinearSVC_classifier_result
    LSVC_results.append(LinearSVC_classifier_result)

    NuSVC_classifier_result = NuSVC_classifier.classify(dict([token, True] for token in clean_tweet))
    print("Nu-Support Vector:", NuSVC_classifier_result)
    tweet_data['Nu-Support Vector'] = NuSVC_classifier_result
    NuSVC_results.append(NuSVC_classifier_result)

    SGDC_classifier_result = SGDC_classifier.classify(dict([token, True] for token in clean_tweet))
    print("Stochastic Gradient Descent:", SGDC_classifier_result)
    tweet_data['Stochastic Gradient Descent'] = SGDC_classifier_result
    SGDC_results.append(SGDC_classifier_result)

# when all the tweets being analysed, get the lis tof results and insert it as  a column in our dataframe
tweet_data['Naive Bayes'] = NB_results
tweet_data['Multinomial Naive Bayes'] = MNB_results
tweet_data['Bernoulli Naive Bayes'] = BNB_results
tweet_data['Logistic Regression'] = LG_results
tweet_data['Linear Support Vector'] = LSVC_results
tweet_data['Nu-Support Vector'] = NuSVC_results
tweet_data['Stochastic Gradient Descent'] = SGDC_results

# here we are printing our results of dataframe and save the outcome of the sentiment analysis module into a new csv
# file
print(tweet_data.to_string())
tweet_data.to_csv("ProcessedData/SentimentAnalysisResults.csv", encoding='utf-8', index=False)
