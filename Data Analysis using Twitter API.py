# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:32:54 2018

@author: Bp_shantam Malgaonka
"""

# Import the required libraries.
import tweepy                        # To consume Twitter's API
import pandas as pd                  # To handle data
import matplotlib.pyplot as plt      # For plotting and visualization:


# Make the graphs prettier
#pd.set_option('display.mpl_style', 'default')

pd.set_option('display.unicode.east_asian_width', True)


consumerKey = 'aChkIToOW3fc7ItXyLmd8vUS6'
consumerSecret = '99UFOywbc6yUYknCdrU6JQiS3UHnxKDeKMcIfZ2lKx9YCPLuI1'

#Use tweepy.OAuthHandler to create an authentication using the given key and secret
auth = tweepy.OAuthHandler(consumer_key=consumerKey, 
    consumer_secret=consumerSecret)

#Connect to the Twitter API using the authentication
api = tweepy.API(auth)

#Perform a basic search query where we search for the'#Oscars2015' in the tweets
result = api.search(q='%23Colgate') #%23 is used to specify '#'

# Print the number of items returned by the search query to verify our query ran. Its 15 by default
len(result)

tweet = result[0] #Get the first tweet in the result

# Analyze the data in one tweet to see what we require
for param in dir(tweet):
#The key names beginning with an '_' are hidden ones and usually not required, so we'll skip them
    if not param.startswith("_"):
        print("%s : %s\n" % (param, eval('tweet.'+param)))
        
        
results = []

#Get the first 5000 items based on the search query
for tweet in tweepy.Cursor(api.search, q='%23Colgate').items(5000):
    results.append(tweet)

# Verify the number of items returned
print(len(results))


# Create a function to convert a given list of tweets into a Pandas DataFrame.
# The DataFrame will consist of only the values, which I think might be useful for analysis...


def toDataFrame(tweets):

    DataSet = pd.DataFrame()

    DataSet['tweetID'] = [tweet.id for tweet in tweets]
    DataSet['tweetText'] = [tweet.text for tweet in tweets]
    DataSet['tweetRetweetCt'] = [tweet.retweet_count for tweet 
    in tweets]
    DataSet['tweetFavoriteCt'] = [tweet.favorite_count for tweet 
    in tweets]
    DataSet['tweetSource'] = [tweet.source for tweet in tweets]
    DataSet['tweetCreated'] = [tweet.created_at for tweet in tweets]


    DataSet['userID'] = [tweet.user.id for tweet in tweets]
    DataSet['userScreen'] = [tweet.user.screen_name for tweet 
    in tweets]
    DataSet['userName'] = [tweet.user.name for tweet in tweets]
    DataSet['userCreateDt'] = [tweet.user.created_at for tweet 
    in tweets]
    DataSet['userDesc'] = [tweet.user.description for tweet in tweets]
    DataSet['userFollowerCt'] = [tweet.user.followers_count for tweet 
    in tweets]
    DataSet['userFriendsCt'] = [tweet.user.friends_count for tweet 
    in tweets]
    DataSet['userLocation'] = [tweet.user.location for tweet in tweets]
    DataSet['userTimezone'] = [tweet.user.time_zone for tweet 
    in tweets]

    return DataSet

#Pass the tweets list to the above function to create a DataFrame
DataSet = toDataFrame(results)

# Let's check the top 5 records in the Data Set
DataSet.head(5)

# Similarly let's check the last 2 records in the Data Set
DataSet.tail(2)

# 'None' is treated as null here, so I'll remove all the records having 'None' in their 'userTimezone' column
#DataSet = DataSet[DataSet.userTimezone.notnull()]

# Let's also check how many records are we left with now
#len(DataSet)

# Count the number of tweets in each User Location and get the first 10
tzs = DataSet['userLocation'].value_counts()[:10]
print (tzs)

# Create a bar-graph figure of the specified size
plt.rcParams['figure.figsize'] = (15, 5)

# Plot the Time Zone data as a bar-graph
tzs.plot(kind='bar')


# Assign labels and title to the graph to make it more presentable
plt.xlabel('Timezones')
plt.ylabel('Tweet Count')
plt.title('Top 10 UserLocations tweeting about #Colgate')
          
plt.savefig('C:/Users/Bp_shantam Malgaonka/Desktop/SHANTAM/Data Analysis using Twitter API and Sentiment Analysis/tzs.jpeg',dpi=400) #to save the file run from line numnber 108





################################################## SENTIMENT ANALYSIS #######################################################

#SENTIMENT ANALYSIS

from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
    
    # We create a column with the result of the analysis:
DataSet['SA'] = ([ analize_sentiment(tweet) for tweet in DataSet['tweetText'] ])

# We display the updated dataframe with the new column:
display(DataSet.head(10))
    
# We construct lists with classified tweets:

pos_tweets = [ tweet for index, tweet in enumerate(DataSet['tweetText']) if DataSet['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(DataSet['tweetText']) if DataSet['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(DataSet['tweetText']) if DataSet['SA'][index] < 0]    
 
# We print percentages:
   
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(DataSet['tweetText'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(DataSet['tweetText'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(DataSet['tweetText'])))
    
#Visualization Part
    
# Count the number of tweets in each SA and get the first 10
SAP = DataSet['SA'].value_counts()[:10]  #SAP=Sentiment Analysis Percentage
print (SAP)
    
# Create a bar-graph figure of the specified size
plt.rcParams['figure.figsize'] = (15, 5)

# Plot the Sentiment Analysis Feedback data as a bar-graph
SAP.plot(kind='bar')
    
# Assign labels and title to the graph to make it more presentable
plt.xlabel('Sentiment Analysis Feedback')
plt.ylabel('Tweet Count')
plt.title('Sentiment Analysis Feedback tweeting about #Colgate')

plt.savefig('C:/Users/Bp_shantam Malgaonka/Desktop/SHANTAM/Data Analysis using Twitter API and Sentiment Analysis/SAP.jpeg',dpi=400) #to save the file run from line numnber 108




#WORD - Cloud
#Python NLTK sentiment analysis

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

#I decided to only do sentiment analysis on this dataset, therfore I dropped the unnecessary colunns, keeping only sentiment and text.

data = DataSet[['tweetText','SA']]

#First of all, splitting the dataset into a training and a testing set. The test set is the 10% of the original dataset.
#For this particular analysis I dropped the neutral tweets, as my goal was to only differentiate positive and negative tweets.

# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)
# Removing neutral sentiments
train = train[train.SA != 0]


#As a next step I separated the Positive and Negative tweets of the training set in order to easily visualize their contained words.
#After that I cleaned the text from hashtags, mentions and links.
#Now they were ready for a WordCloud visualization which shows only the most emphatic words of the Positive and Negative tweets.


train_pos = train[ train['SA'] == 1]
train_pos = train_pos['tweetText']
train_neg = train[ train['SA'] == -1]
train_neg = train_neg['tweetText']


def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_draw(train_pos,'white')
print("Negative words")
wordcloud_draw(train_neg)