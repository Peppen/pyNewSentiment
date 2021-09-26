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



def download(folder, username):
    with open(folder + '/' + username + '.json', 'w') as f:
        for tweet in tweepy.Cursor(auth_api.user_timeline, id=username).items(100):
            f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')


if __name__ == '__main__':
    download('json', 'WashTimes')
