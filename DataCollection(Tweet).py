import time
start_counting = time.time()
import csv
import tweepy

consumer_key = 'oSYrCa75rIA7E3a8LSbBvJxPe'
consumer_secret = 'GJAf741pOWPsRD6ZJ0e8QlYrnCDcPIbE579eNyZOaY9Fe9fByh'
access_token = '2987300974-18sYqXpFrlIPVgj84L9dBwoioFNWjzhaz3ubzVD'
access_token_secret = 'mAfnM2p6SyLpkDuwBFobYi2f2HJHtf4zN9uSqTI84PlVZ'

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
                           since_id=1240322076734610000,
                           count=100,
                           lang="en").items():
    # print output of the section and save the outcomes to a csv file.
    print(tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.id, tweet.created_at, tweet.text.encode('utf-8')])

print('It took {0:0.1f} seconds'.format(time.time() - start_counting))