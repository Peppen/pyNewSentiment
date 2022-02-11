import jsonpickle
import tweepy
from tweepy import OAuthHandler, API

consumer_key = "sVZIgPw8ngb43yUVJhiEjIMjj"
consumer_secret = "qnrmO4zMRzCbLdSVBuK94hGCgkGbmRw9wCyh83rsXLc9sSWTW4"
access_token = "1324015425546522631-TggWens5XxMqHUzc2rD6DHVMx1UGgg"
access_token_secret = "lEf3UonZRKQda7LD3Jcgc5NQNmMXIToVpOcheICtSwBlf"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth)


# Downloading Tweets
def download():
    for tweet in tweepy.Cursor(auth_api.user_timeline, id='BBCNews').items(100):
        print(tweet)




