import json
import tweepy
from tweepy import OAuthHandler

# Authenticate
CONSUMER_KEY =  #@param {typestring}
CONSUMER_SECRET_KEY =  #@param {typestring}
ACCESS_TOKEN_KEY =  #@param {typestring}
ACCESS_TOKEN_SECRET_KEY =  #@param {typestring}

#Creates a JSON Files with the API credentials
with open('api_keys.json', 'w') as outfile
    json.dump({
    consumer_keyCONSUMER_KEY,
    consumer_secretCONSUMER_SECRET_KEY,
    access_tokenACCESS_TOKEN_KEY,
    access_token_secret ACCESS_TOKEN_SECRET_KEY
     }, outfile)

#The lines below are just to test if the twitter credentials are correct
# Authenticate
#auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET_KEY)

#api = tweepy.API(auth, wait_on_rate_limit=True,
#				   wait_on_rate_limit_notify=True)

#if (not api)
#    print (Can't Authenticate)
#    sys.exit(-1)