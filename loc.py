#Group-16
#Kartik
#Roma
#Vidyalakshmi

#Code by Prof.Gene Moo Lee, in section INSY-5378-001-2017-Spring

#!pip install Twython
from twython import TwythonStreamer
import sys
import json
import time

tweets = []

class MyStreamer(TwythonStreamer):
    '''our own subclass of TwythonStremer'''
  
    # overriding
    try:
        def on_success(self, data):
            if ('lang' in data and data['lang'] == 'en'):#('Trump' in data['lang'] or 'trump' in data['lang']):
                #put the name of the states under observation
                if ((data['user'])['location']) != None and ('UT' in ((data['user'])['location']) or 'Utah' in ((data['user'])['location']) or 'AZ' in ((data['user'])['location']) or 'Arizona' in ((data['user'])['location'])): #((data['user'])['location']).find('TX'):
                    tweets.append(data)#.encode("utf-8")
                    print 'received tweet #', len(tweets), data['text'].encode("utf-8")
                else:
                    print 'irrelevant'    

            if len(tweets) >= 1000:
                self.store_json()
                self.disconnect()
            #if len(tweets)%5 == 0:
            #    time.sleep(10)
    except:
        pass

    # overriding
    try:
        def on_error(self, status_code, data):
            print status_code, data
            self.disconnect()
    except:
        pass

    def store_json(self):
        #change the 'UT_AZ' value to the states under observation
        with open('tweet_stream_{}_1000.json'.format('UT_AZ'), 'w') as f:
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

    stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    #if len(sys.argv) > 1:
    #    keyword = sys.argv[1]
    #else:
    #    keyword = 'trump'
    
#To overcome ChunckedEncodingError-IncompleteRead
    counter=0
    while(len(tweets)<1000):
        counter+=1
        try:
            if(counter%3!=0):
                #Taken from- https://dev.twitter.com/streaming/overview/request-parameters#track
                #Locations taken from http://boundingbox.klokantech.com/
                #change the location values & run again to get another state or combination of states.
                stream.statuses.filter(location='-114.05,37.0,-109.04,42.0,-114.82,31.33,-109.05,37.0',track='trump')
            else:
                #Taken from - http://stackoverflow.com/questions/510348/how-can-i-make-a-time-delay-in-python
                #to overcome error: 420 Easy there Turbo, too many requests recently 
                time.sleep(10)
        except:
            continue
