#Group-16
#Kartik
#Roma
#Vidyalakshmi

#Code by Gene Moo Lee, in section INSY-5378-001-2017-Spring

#!pip install Twython
#!pip install Twython
from twython import TwythonStreamer
import sys
import json
import time
import warnings


tweets = []

class MyStreamer(TwythonStreamer):
    '''our own subclass of TwythonStremer'''
    
    # overriding
    
    try:
    	def on_success(self, data):
	        if 'lang' in data and data['lang'] == 'en':
	            tweets.append(data)#.encode("utf-8")
	            print 'received tweet #', len(tweets), data['text'].encode("utf-8")
	                        

	        if len(tweets) >= 500:
	            self.store_json()
	            self.disconnect()
#	        if len(tweets)%100 == 0:
#	            time.sleep(10)
    except:
    	while('True'):
    		continue
                
        

    # overriding
    try:
    	def on_error(self, status_code, data):
        	while(len(tweets)>=10000):
        		break
    except:
    	pass

    def store_json(self):
        with open('tweet_stream_trump_20.json', 'w') as f:
            json.dump(tweets, f, indent=4)


if __name__ == '__main__':

    with open('vidya_twitter_credentials.json', 'r') as f:
    #with open('../../../JG_Ch09_Getting_Data/04_api/gene_twitter_credentials.json', 'r') as f:
        credentials = json.load(f)

    # create your own app to get consumer key and secret
    CONSUMER_KEY = credentials['CONSUMER_KEY']
    CONSUMER_SECRET = credentials['CONSUMER_SECRET']
    ACCESS_TOKEN = credentials['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = credentials['ACCESS_TOKEN_SECRET']
    #CONSUMER_KEY = 'KdKea6waaXUm1M6pQ9WwRUNls'
    #CONSUMER_SECRET = 'Z1kddgvKg5mv2IBLUjkeKfoRiUV51VtRwbpCvxSZnX8qb3PrQk'
    #ACCESS_TOKEN = '837801751147397120-u2QUknSkiISWU7R8vAflJ6H7974bp3G'
    #ACCESS_TOKEN_SECRET = 'INWeopIeF0ftndvrBHXFO37KWpPFF5GzR0TOwYEgvtn0a'

    stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    if len(sys.argv) > 1:
        keyword = sys.argv[1]
    else:
        keyword = 'trump'

    stream.statuses.filter(track='trump',stall_warnings=True)
