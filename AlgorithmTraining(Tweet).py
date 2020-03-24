import pickle
import random
import re
import string

import nltk
from nltk import FreqDist, classify, NaiveBayesClassifier, SklearnClassifier
from nltk.corpus import twitter_samples, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC


# method to remove noise from the dataframe to a word list
# using regular expressions clean the sentences
# below sections specifies the parts of th nltk speech tagging method and what going to be used for word tokenizaiton
# converts words into dictionary words


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

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


# takes a list of words and goes through every single word individually
# similar can be applied using sentence tokenizer but this could effect accuracy
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


# this method is used for the training purposes and display the most common features

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


# main method without being defined, as we want this module to run main method always when is started
if __name__ == "__main__":

    # using natural language packages we can use sample tweets already gathered to train and test our data
    # this will save the time  as we do not need to download this data ourselves
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
    # specify what dictionary language we are considering to create a dictionary words
    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    # get sample tweets through remove noise methods adn convert them into words and put into list of positive or
    # negative
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    # running through the list of words from sample tweets and displays the most common features of the list,
    # this will give us a visual idea and we can interpret it towards the accuracy of the algorithms
    freq_dist_pos = FreqDist(all_pos_words)
    print("The most common 10 words:")
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    # shuffle data set to avoid bayes train and test set, this will ensure that we run aour algo on random data set
    # rather than training it on only positive tweets and test against negative. This will not give the accurate and
    # credible output
    random.shuffle(dataset)
    print("Dataset size", len(dataset), "\n")

    # here we specify size otf training and tes. It is important and recommended to train against less data that we
    # train size of our data set is 10k, which based on research is recommended for the small to medium size of the
    # problem
    train_data = dataset[:3000]
    test_data = dataset[7000:]

    # from this section we are running our project algorithms against 7 classifiers that will display theirs individual
    # accuracy. TO further speed up the process of loading the trained algorithms, we are saving the outputs of these
    # tests into pickle file so we can load them in later stages when required
    NB_classifier = NaiveBayesClassifier.train(train_data)
    print("Naive Bayes accuracy is:", (classify.accuracy(NB_classifier, test_data)) * 100, "%")
    save_classifier = open("TrainedAlgorithms/NB_classifier.pickle", "wb")
    pickle.dump(NB_classifier, save_classifier)
    save_classifier.close()

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(train_data)
    print("Multinomial Naive Bayes accuracy is:", (nltk.classify.accuracy(MNB_classifier, test_data)) * 100, "%")
    save_classifier = open("TrainedAlgorithms/MNB_classifier.pickle", "wb")
    pickle.dump(MNB_classifier, save_classifier)
    save_classifier.close()

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(train_data)
    print("Bernoulli Naive Bayes accuracy is:", (nltk.classify.accuracy(BernoulliNB_classifier, test_data)) * 100, "%")
    save_classifier = open("TrainedAlgorithms/BernoulliNB_classifier.pickle", "wb")
    pickle.dump(BernoulliNB_classifier, save_classifier)
    save_classifier.close()

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(train_data)
    print("Logistic Regression accuracy is:",
          (nltk.classify.accuracy(LogisticRegression_classifier, test_data)) * 100, "%")
    save_classifier = open("TrainedAlgorithms/LogisticRegression_classifier.pickle", "wb")
    pickle.dump(LogisticRegression_classifier, save_classifier)
    save_classifier.close()

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(train_data)
    print("Linear Support Vector Classification accuracy is:",
          (nltk.classify.accuracy(LinearSVC_classifier, test_data)) * 100, "%")
    save_classifier = open("TrainedAlgorithms/LinearSVC_classifier.pickle", "wb")
    pickle.dump(LinearSVC_classifier, save_classifier)
    save_classifier.close()

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(train_data)
    print("Nu-Support Vector Classification accuracy is:", (nltk.classify.accuracy(NuSVC_classifier, test_data)) * 100,
          "%")
    save_classifier = open("TrainedAlgorithms/NuSVC_classifier.pickle", "wb")
    pickle.dump(NuSVC_classifier, save_classifier)
    save_classifier.close()

    SGDC_classifier = SklearnClassifier(SGDClassifier())
    SGDC_classifier.train(train_data)
    print("Stochastic Gradient Descent accuracy is:", nltk.classify.accuracy(SGDC_classifier, test_data) * 100, "%\n")
    save_classifier = open("TrainedAlgorithms/SGDC_classifier.pickle", "wb")
    pickle.dump(SGDC_classifier, save_classifier)
    save_classifier.close()
