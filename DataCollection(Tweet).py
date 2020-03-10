import csv
import tweepy

consumer_key = 'xyz'
consumer_secret = 'xyz'
access_token = 'xyz-xyz'
access_token_secret = 'xyz'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Open/Create a file to append data
csvFile = open('OutputData/CryptoTweetDataNoRetweetsID.csv', 'a')

# Use csv Writer
csvWriter = csv.writer(csvFile)

# specify the search titles, result types data and exclude retweets.

for tweet in tweepy.Cursor(api.search, q="bitcoin OR cryptocurrency OR cryptomarket -filter:retweets",
                           result_type="mixed",
                           since_id=1220728293638440000,
                           count=200,
                           lang="en").items():
    # print output of the section and save the outcomes to a csv file.
    print(tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.id, tweet.created_at, tweet.text.encode('utf-8')])
