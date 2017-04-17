#Group-16
#Kartik
#Roma
#Vidyalakshmi

#Code by Prof.Gene Moo Lee, in section INSY-5378-001-2017-Spring


from __future__ import division, print_function
import json
import nltk
import string
from textblob import TextBlob
import numpy as np

#pip install wordcloud
#pip install cloud

#read whole tweetfile as it is (working)
with open('tweet_stream_CA_WA_1000.json') as f:
	data = f.read()
#print type(data)-str

#Taken from- http://stackoverflow.com/questions/20199126/reading-json-from-a-file
values = json.loads(data)

#print(len(values))

#collecting only tweets (working)
tweets=[]
for x in range(0,len(values)):
	#if 'trump' in (values[x])['text'] or 'Trump' in (values[x])['text']:
	tweets.append((((values[x])['text']).lower()).encode('ascii','ignore'))


#print(len(tweets))

#Cleaning the punctuation, digits and stopwords (Working)
p = string.punctuation
d = string.digits
table_p = string.maketrans(p, len(p) * " ")
table_d = string.maketrans(d, len(d) * " ")
tweetsPD=[]

for i in range(0,len(tweets)):
	tweetsPD.append((tweets[i].translate(table_p)).translate(table_d))
#print tweetsPD

stopwords = nltk.corpus.stopwords.words('english')

added_stopwards=[u'trump', u'trumps',u'donald',u'https',u'president',u'co',u'rt',u'today',u'well',u'want',u'said',u'going',u'oh',u'done',u'dem',u'need',u'tweets',u'he',u'so',u'everybody',u'for',u'okay',u'ok',u'at',u'by',u'to',u'under',u'see',u'know',u'tree',u'on',u'line',u'over',u'every',u'being',u'as',u'same',u'running',u'got',u'be',u'wrote',u'about',u'she',u'loser',u'among',u'most',u'after',u'foot',u'yes',u'very',u'major',u'think',u'ass',u'just']
for i in range(0,len(added_stopwards)):
	stopwords.append(added_stopwards[i])
#tweetsPD=['hello how are you','what are you']
final=[]
for i in range(0,len(tweetsPD)):
	tweetsPDS = []
	for t in tweetsPD[i].split():
	    if t not in stopwords and len(t) > 1:
	        tweetsPDS.append(t)
#
#Joining list taken from- http://stackoverflow.com/questions/493819/python-join-why-is-it-string-joinlist-instead-of-list-joinstring
	final.append(' '.join(tweetsPDS))
#print type(final[0]) 
#print tweetsPDS[0]
#Sentiment Analysis (wroking)
tb_pos=[]
sub_list = []
pol_list = []
#print(final)
for i in range(0,len(tweetsPD)):
	tb_pos = TextBlob(final[i])
	#print tb_pos
	#print(tb_pos.sentiment)
	#print(tb_pos.polarity)
	#print(tb_pos.subjectivity)
	sub_list.append(tb_pos.sentiment.subjectivity)
	pol_list.append(tb_pos.sentiment.polarity)
#print type(tb_pos)-textblob

#ploting polarity and subjectivity in histogram (working)
import matplotlib.pyplot as plt

plt.hist(sub_list, bins=10) #, normed=1, alpha=0.75)
plt.xlabel('subjectivity score')
plt.ylabel('sentence count')
plt.grid(True)
plt.savefig('subjectivity.pdf')
plt.show()

plt.hist(pol_list, bins=20) #, normed=1, alpha=0.75)
plt.xlabel('polarity score')
plt.ylabel('sentence count')
plt.grid(True)
plt.savefig('polarity.pdf')
plt.show()

avg_sub=sum(sub_list)/float(len(sub_list))
print('The average subjectivity score: {}'.format(avg_sub))
avg_pol=sum(pol_list)/float(len(pol_list))
print('The average polarity score: {}'.format(avg_pol))


#create wordcloud
from wordcloud import WordCloud

#stemming (working)
from nltk.stem.lancaster import LancasterStemmer
ls = LancasterStemmer()

#Joining list taken from- http://stackoverflow.com/questions/493819/python-join-why-is-it-string-joinlist-instead-of-list-joinstring
text=' '.join(final)
#print text
stemmed=[]
for word in text.split():
    stemmed.append(ls.stem(word))

#print stemmed

#Word Cloud (not working)
wordcloud = WordCloud(max_font_size=40).generate(text)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#topic modeling
#use may be final list as it is encoded
#Vectorize the text and
#Make pairwise document distance based on TF-IDF
#check unique words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
dtm = vectorizer.fit_transform(final)
#print(dtm.shape)
vocab = vectorizer.get_feature_names() # list of unique vocab, we will use this later
print(len(vocab), '# of unique words')
#print vocab[-10:]
#print vocab[:10]

#NMF Decomposition using term-document matrix
from sklearn import decomposition

#print 'num of documents, num of unique words'
#print dtm.shape

num_topics = 5

clf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)
#print(num_topics, clf.reconstruction_err_)


topic_words = []
num_top_words = 5
for topic in clf.components_:
    #print topic.shape, topic[:5]
    word_idx = np.argsort(topic)[::-1][0:num_top_words] # get indexes with highest weights
    #print 'top indexes', word_idx
    topic_words.append([vocab[i] for i in word_idx])
    #print topic_words[-1]
    #print

print('\n\nTopics by Non-negative Matrix Factorization Model')  
    
    
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))

print('\n\n')
#Latent Dirichlet Allocation (LDA) with Gensim
#1
from gensim import corpora, models, similarities, matutils
import re
#import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#2
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore'''

#from nltk.corpus import inaugural
#print(type(inaugural.fileid()))
#names = []
docs = []

#for fileid in inaugural.fileids():
 #   names.append(fileid)
 #   docs.append(inaugural.words(fileid))
#print(type(names))
for i in range(len(final)):
	a=nltk.word_tokenize(final[i])
	docs.append((a))#.encode('ascii','ignore'))
#print(len(names), 'documents in the corpus')
#print(names[:10])
#print(docs[0])

#print(docs[0])

#3
from gensim import corpora
dic = corpora.Dictionary(docs)
#print(dic)

#4
corpus = [dic.doc2bow(text) for text in docs]
#print(type(corpus), len(corpus))

#5
#for corp in corpus:
#    print(len(corp), corp[:10])

#6
from gensim import models
tfidf = models.TfidfModel(corpus)
#print(type(tfidf))

#7
corpus_tfidf = tfidf[corpus]
#print(type(corpus_tfidf))

#8
NUM_TOPICS = 5
model = models.ldamodel.LdaModel(corpus_tfidf, 
                                 num_topics=NUM_TOPICS, 
                                 id2word=dic, 
                                 update_every=1, 
                                 passes=100)

#9
print("\n\n\nTopics by Latent Dirichlet Allocation model")
topics_found = model.print_topics(20)
counter = 1
for t in topics_found:
    print("Topic #{} {}".format(counter, t))
    counter += 1

'''
#10
from gensim import models
NUM_TOPICS = 5
model = models.lsimodel.LsiModel(corpus_tfidf,
                                 id2word=dic,
                                 num_topics=NUM_TOPICS
                                )


#11
model.print_topics()

#12
'''