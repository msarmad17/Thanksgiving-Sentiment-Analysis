
# coding: utf-8

# In[16]:


import sys
import requests
import time
import json
from datetime import datetime
from requests_oauthlib import OAuth1

credentials = {
    'CONSUMER_KEY': '',
    'CONSUMER_SECRET': '',
    'TOKEN_KEY': '',
    'TOKEN_SECRET': '',
}

def authenticate(credentials):
    try:
        oauth = OAuth1(client_key=credentials['CONSUMER_KEY'],
                      client_secret=credentials['CONSUMER_SECRET'],
                      resource_owner_key=credentials['TOKEN_KEY'],
                      resource_owner_secret=credentials['TOKEN_SECRET'],
                      signature_type='auth_header')
        client = requests.session()
        client.auth = oauth
        return client
    except (KeyError, TypeError):
        print('Error setting auth credentials.')
        raise



MAX_TWEETS = 500

# API endpoint
url = 'https://stream.twitter.com/1.1/statuses/filter.json'

# gather tweets from Manhattan
client = authenticate(credentials)
response = client.get(url, stream=True, params={'locations': '-74,40,-73,41'})

if response.ok:
    f = open("tweets.json","wb")
    num_tweets = 0
    try:
        for line in response.iter_lines():
            # stop after reaching MAX_TWEETS
            if num_tweets == MAX_TWEETS:
                break
            # Twitter sends empty lines to keep the connection alive. We need to filter those.
            if line:
                f.write(line + b'\n')
                num_tweets += 1
                print(".", end='', flush=True)
    except KeyboardInterrupt:
        # User pressed the 'Stop' button
        print()
        print('Data collection interrupted by user!')
    finally:
        # Cleanup -- close file and report number of tweets collected 
        f.close()
        print()
        print('Collected {} tweets.'.format(num_tweets))
else:
    print('Connection failed with status: {}'.format(response.status_code))


# laod tweets
tweets = []

with open('tweets.json') as f:
    for line in f:
        obj = json.loads(line)
        tweets.append(obj)


tweets = []

for l in statuses:
    for s in l:
        tweets.append(s)
        
url = 'https://api.twitter.com/1.1/statuses/user_timeline.json?'



# remove multiple tweets from each user, keeping one tweet from each user
for t in tweets:
    for u in tweets:
        if t['id'] != u['id'] and t['user']['id'] == u['user']['id']:
            tweets.remove(u)


# gather timelines for users in the tweets list

u_timelines = []

for t in tweets:
    url = 'https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name=' + t['user']['screen_name'] + '&since_id=1064482777054220289&count=200'
    req = client.get(url)
    
    if req.status_code == 200:
        u_timelines.append(req.json())
        d = len(u_timelines) - 1
        
    while len(req.json()) == 200:
            url = 'https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name=' + t['user']['screen_name'] + '&max_id=' + u_timelines[d][len(u_timelines[d]) - 1]['id_str'] + '&count=200'
            req = client.get(url)
            
            if req.status_code == 200 and len(req.json()) != 0:
                del u_timelines[d][len(u_timelines[d]) - 1]
                u_timelines[d].extend(req.json())


for u in u_timelines:
    for t in u:
        a = int(t['created_at'][8:10])
        if a < 19:
            u.remove(t)





import json
from pandas.io.json import json_normalize

tweet_lists = [dict(j) for i in u_timelines for j in i]
dfItem = json_normalize(tweet_lists, errors='ignore')

dfItem['created_at'] = pd.to_datetime(dfItem['created_at'],dayfirst=True, errors='coerce')
#dfItem.to_csv('all_tweets',header=True, index=False, encoding='utf-8')


dd = []

for i in tweet_lists:
        dict_temp = {}
        dict_temp['screen_name'] = i['user']['screen_name']
        dict_temp['created_at'] = i['created_at']
        dict_temp['id'] = i['id']
        dict_temp['text'] = i['text']

        dd.append(dict_temp)

dfItem = json_normalize(dd, errors='ignore')


# get all tweets from all timelines as pandas dataframe

import pandas as pd

dfItem['created_at'] = pd.to_datetime(dfItem['created_at'],dayfirst=True, errors='coerce')

dfItem.head()


# filter out tweets before the 19th of November

ddd =  dfItem[dfItem.created_at>='2018-11-19']





#  create separate dataframe for each date of the Thanksgiving week, each containing the corresponding tweets
df19 = ddd[ddd.created_at<'2018-11-20'

tmp = ddd[ddd.created_at<'2018-11-21']
df20 = tmp[tmp.created_at>'2018-11-20']

tmp = ddd[ddd.created_at<'2018-11-22']
df21 = tmp[tmp.created_at>'2018-11-21']

tmp = ddd[ddd.created_at<'2018-11-23']
df22 = tmp[tmp.created_at>'2018-11-22']

tmp = ddd[ddd.created_at<'2018-11-24']
df23 = tmp[tmp.created_at>'2018-11-23']

tmp = ddd[ddd.created_at<'2018-11-25']
df24 = tmp[tmp.created_at>'2018-11-24']

df25 = ddd[ddd.created_at>'2018-11-25']



# import nltk, define sentiment analysis functon, and conduct analysis on each dataframe (each date of the week)

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def nltk_sentiment(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score


mood19 = []

for i in range(df19.shape[0]):
    mood19.append(nltk_sentiment(df19.iloc[i]['text']))


mood20 = []

for i in range(df20.shape[0]):
    mood20.append(nltk_sentiment(df20.iloc[i]['text']))


mood21 = []

for i in range(df21.shape[0]):
    mood21.append(nltk_sentiment(df21.iloc[i]['text']))


mood22 = []

for i in range(df22.shape[0]):
    mood22.append(nltk_sentiment(df22.iloc[i]['text']))


mood23 = []

for i in range(df23.shape[0]):
    mood23.append(nltk_sentiment(df23.iloc[i]['text']))


mood24 = []

for i in range(df24.shape[0]):
    mood24.append(nltk_sentiment(df24.iloc[i]['text']))


mood25 = []

for i in range(df25.shape[0]):
    mood25.append(nltk_sentiment(df25.iloc[i]['text']))



# calculate average positivity and negativity for each date

av_pos_19 = 0
n_pos_19 = 0
av_neg_19 = 0
n_neg_19 = 0

for i in mood19:
    if i['pos'] != 0.0:
        av_pos_19 += i['pos']
        n_pos_19 += 1
    if i['neg'] != 0.0:
        av_neg_19 += i['neg']
        n_neg_19 += 1

av_pos_19 = av_pos_19 / n_pos_19
av_neg_19 = av_neg_19 / n_neg_19


av_pos_20 = 0
n_pos_20 = 0
av_neg_20 = 0
n_neg_20 = 0

for i in mood20:
    if i['pos'] != 0.0:
        av_pos_20 += i['pos']
        n_pos_20 += 1
    if i['neg'] != 0.0:
        av_neg_20 += i['neg']
        n_neg_20 += 1

av_pos_20 = av_pos_20 / n_pos_20
av_neg_20 = av_neg_20 / n_neg_20


av_pos_21 = 0
n_pos_21 = 0
av_neg_21 = 0
n_neg_21 = 0

for i in mood21:
    if i['pos'] != 0.0:
        av_pos_21 += i['pos']
        n_pos_21 += 1
    if i['neg'] != 0.0:
        av_neg_21 += i['neg']
        n_neg_21 += 1

av_pos_21 = av_pos_21 / n_pos_21
av_neg_21 = av_neg_21 / n_neg_21


av_pos_22 = 0
n_pos_22 = 0
av_neg_22 = 0
n_neg_22 = 0

for i in mood22:
    if i['pos'] != 0.0:
        av_pos_22 += i['pos']
        n_pos_22 += 1
    if i['neg'] != 0.0:
        av_neg_22 += i['neg']
        n_neg_22 += 1

av_pos_22 = av_pos_22 / n_pos_22
av_neg_22 = av_neg_22 / n_neg_22


av_pos_23 = 0
n_pos_23 = 0
av_neg_23 = 0
n_neg_23 = 0

for i in mood23:
    if i['pos'] != 0.0:
        av_pos_23 += i['pos']
        n_pos_23 += 1
    if i['neg'] != 0.0:
        av_neg_23 += i['neg']
        n_neg_23 += 1

av_pos_23 = av_pos_23 / n_pos_23
av_neg_23 = av_neg_23 / n_neg_23


av_pos_24 = 0
n_pos_24 = 0
av_neg_24 = 0
n_neg_24 = 0

for i in mood24:
    if i['pos'] != 0.0:
        av_pos_24 += i['pos']
        n_pos_24 += 1
    if i['neg'] != 0.0:
        av_neg_24 += i['neg']
        n_neg_24 += 1

av_pos_24 = av_pos_24 / n_pos_24
av_neg_24 = av_neg_24 / n_neg_24


av_pos_25 = 0
n_pos_25 = 0
av_neg_25 = 0
n_neg_25 = 0

for i in mood25:
    if i['pos'] != 0.0:
        av_pos_25 += i['pos']
        n_pos_25 += 1
    if i['neg'] != 0.0:
        av_neg_25 += i['neg']
        n_neg_25 += 1

av_pos_25 = av_pos_25 / n_pos_25
av_neg_25 = av_neg_25 / n_neg_25





# plot average positivity and negativity for each date of the week

import matplotlib.pyplot as plt 
  
# line 1 points 
x1 = [19.0,20.0,21.0,22.0,23.0,24.0,25.0] 
y1 = [av_pos_19, av_pos_20, av_pos_21, av_pos_22, av_pos_23, av_pos_24, av_pos_25]
# plotting the line 1 points  
plt.plot(x1, y1, label = "Positive Sentiment") 
  
# line 2 points 
x2 = [19.0,20.0,21.0,22.0,23.0,24.0,25.0]
y2 = [av_neg_19, av_neg_20, av_neg_21, av_neg_22, av_neg_23, av_neg_24, av_neg_25] 
# plotting the line 2 points  
plt.plot(x2, y2, label = "Negative Sentiment") 
  
# naming the x axis 
plt.xlabel('Day of November') 
# naming the y axis 
plt.ylabel('Sentiment Intensity') 
# giving a title to my graph 
plt.title('Sentiment Variation During Thanksgiving Week') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 





graph_tweets = [dict(j) for i in u_timelines for j in i]

graph_dict = []

for i in graph_tweets:
        dict_tmp = {}
        dict_tmp['screen_name'] = i['user']['screen_name']
        dict_tmp['hashtags'] = i['entities']['hashtags']
        dict_tmp['created_at'] = i['created_at']
        dict_tmp['id'] = i['id']
        dict_tmp['text'] = i['text']

        graph_dict.append(dict_tmp)

graph_df = json_normalize(graph_dict, errors='ignore')


graph_df['created_at'] = pd.to_datetime(graph_df['created_at'],dayfirst=True, errors='coerce')

trim_df =  graph_df[graph_df.created_at>='2018-11-19']


# recreate dataframes for each date, this time INCLUDING HASHTAGS as a separate attribute as well for each tweet entity

gph19 = trim_df[trim_df.created_at<'2018-11-20']

temp = trim_df[trim_df.created_at<'2018-11-21']
gph20 = temp[temp.created_at>'2018-11-20']

temp = trim_df[trim_df.created_at<'2018-11-22']
gph21 = temp[temp.created_at>'2018-11-21']

temp = trim_df[trim_df.created_at<'2018-11-23']
gph22 = temp[temp.created_at>'2018-11-22']

temp = trim_df[trim_df.created_at<'2018-11-24']
gph23 = temp[temp.created_at>'2018-11-23']

temp = trim_df[trim_df.created_at<'2018-11-25']
gph24 = temp[temp.created_at>'2018-11-24']

gph25 = trim_df[trim_df.created_at>'2018-11-25']



import networkx as nx

# create bipartite graphs for each date, with the disjont sets of nodes being the USERS and the HASHTAGS,
# and edge weights determining the number of times a hashtag was used by a user

cd = {}

for i in range(gph19.shape[0]):
    try:
        user_name = gph19.iloc[i]['screen_name']
    except KeyError:
        continue
    try:
        user_dict = cd[user_name]
    except:
        cd[user_name] = {}
        user_dict = cd[user_name]
    for hashtag_dict in gph19.iloc[i]['hashtags']:
        hashtag = '#' + hashtag_dict['text']
        try:
            user_dict[hashtag]['weight'] += 1
        except KeyError:
            user_dict[hashtag] = {'weight': 1}
                
G19 = nx.from_dict_of_dicts(cd)


ef = {}

for i in range(gph20.shape[0]):
    try:
        user_name = gph20.iloc[i]['screen_name']
    except KeyError:
        continue
    try:
        user_dict = ef[user_name]
    except:
        ef[user_name] = {}
        user_dict = ef[user_name]
    for hashtag_dict in gph20.iloc[i]['hashtags']:
        hashtag = '#' + hashtag_dict['text']
        try:
            user_dict[hashtag]['weight'] += 1
        except KeyError:
            user_dict[hashtag] = {'weight': 1}
                
G20 = nx.from_dict_of_dicts(ef)


gh = {}

for i in range(gph21.shape[0]):
    try:
        user_name = gph21.iloc[i]['screen_name']
    except KeyError:
        continue
    try:
        user_dict = gh[user_name]
    except:
        gh[user_name] = {}
        user_dict = gh[user_name]
    for hashtag_dict in gph21.iloc[i]['hashtags']:
        hashtag = '#' + hashtag_dict['text']
        try:
            user_dict[hashtag]['weight'] += 1
        except KeyError:
            user_dict[hashtag] = {'weight': 1}
                
G21 = nx.from_dict_of_dicts(gh)


ij = {}

for i in range(gph22.shape[0]):
    try:
        user_name = gph22.iloc[i]['screen_name']
    except KeyError:
        continue
    try:
        user_dict = ij[user_name]
    except:
        ij[user_name] = {}
        user_dict = ij[user_name]
    for hashtag_dict in gph22.iloc[i]['hashtags']:
        hashtag = '#' + hashtag_dict['text']
        try:
            user_dict[hashtag]['weight'] += 1
        except KeyError:
            user_dict[hashtag] = {'weight': 1}
                
G22 = nx.from_dict_of_dicts(ij)


kl = {}

for i in range(gph23.shape[0]):
    try:
        user_name = gph23.iloc[i]['screen_name']
    except KeyError:
        continue
    try:
        user_dict = kl[user_name]
    except:
        kl[user_name] = {}
        user_dict = kl[user_name]
    for hashtag_dict in gph23.iloc[i]['hashtags']:
        hashtag = '#' + hashtag_dict['text']
        try:
            user_dict[hashtag]['weight'] += 1
        except KeyError:
            user_dict[hashtag] = {'weight': 1}
                
G23 = nx.from_dict_of_dicts(kl)


mn = {}

for i in range(gph24.shape[0]):
    try:
        user_name = gph24.iloc[i]['screen_name']
    except KeyError:
        continue
    try:
        user_dict = mn[user_name]
    except:
        mn[user_name] = {}
        user_dict = mn[user_name]
    for hashtag_dict in gph24.iloc[i]['hashtags']:
        hashtag = '#' + hashtag_dict['text']
        try:
            user_dict[hashtag]['weight'] += 1
        except KeyError:
            user_dict[hashtag] = {'weight': 1}
                
G24 = nx.from_dict_of_dicts(mn)


op = {}

for i in range(gph25.shape[0]):
    try:
        user_name = gph25.iloc[i]['screen_name']
    except KeyError:
        continue
    try:
        user_dict = op[user_name]
    except:
        op[user_name] = {}
        user_dict = op[user_name]
    for hashtag_dict in gph25.iloc[i]['hashtags']:
        hashtag = '#' + hashtag_dict['text']
        try:
            user_dict[hashtag]['weight'] += 1
        except KeyError:
            user_dict[hashtag] = {'weight': 1}
                
G25 = nx.from_dict_of_dicts(op)



print(G19)
print("The number of nodes is: {}".format(G19.number_of_nodes()))
print("The number of edges is: {}".format(G19.number_of_edges()))
print("The Graph is bipartite: {}".format(nx.is_bipartite(G19)))
print("The Graph is directed: {}".format(nx.is_directed(G19)))


# add all edge weights for each hashtag that fits the hashtag criteria whereby they are related to the holiday season
# Criteria: Holiday season terms such as Thanksgiving, holiday, and turkey, with tolerance for title case

deg19 = G19.degree()
t19 = 0
for i in deg19:
    if i[0][:6] == '#Thank' or i[0][:6] == '#thank' or i[0][:5] == '#Holi' or i[0][:5] == '#holi' or i[0][:7] == '#Turkey' or i[0][:7] == '#turkey':
        t19 += i[1]


deg20 = G20.degree()
t20 = 0
for i in deg20:
    if i[0][:6] == '#Thank' or i[0][:6] == '#thank' or i[0][:5] == '#Holi' or i[0][:5] == '#holi' or i[0][:7] == '#Turkey' or i[0][:7] == '#turkey':
        t20 += i[1]


deg21 = G21.degree()
t21 = 0
for i in deg21:
    if i[0][:6] == '#Thank' or i[0][:6] == '#thank' or i[0][:5] == '#Holi' or i[0][:5] == '#holi' or i[0][:7] == '#Turkey' or i[0][:7] == '#turkey':
        t21 += i[1]


deg22 = G22.degree()
t22 = 0
for i in deg22:
    if i[0][:6] == '#Thank' or i[0][:6] == '#thank' or i[0][:5] == '#Holi' or i[0][:5] == '#holi' or i[0][:7] == '#Turkey' or i[0][:7] == '#turkey':
        t22 += i[1]


deg23 = G23.degree()
t23 = 0
for i in deg23:
    if i[0][:6] == '#Thank' or i[0][:6] == '#thank' or i[0][:5] == '#Holi' or i[0][:5] == '#holi' or i[0][:7] == '#Turkey' or i[0][:7] == '#turkey':
        t23 += i[1]


deg24 = G24.degree()
t24 = 0
for i in deg24:
    if i[0][:6] == '#Thank' or i[0][:6] == '#thank' or i[0][:5] == '#Holi' or i[0][:5] == '#holi' or i[0][:7] == '#Turkey' or i[0][:7] == '#turkey':
        t24 += i[1]


deg25 = G25.degree()
t25 = 0
for i in deg25:
    if i[0][:6] == '#Thank' or i[0][:6] == '#thank' or i[0][:5] == '#Holi' or i[0][:5] == '#holi' or i[0][:7] == '#Turkey' or i[0][:7] == '#turkey':
        t25 += i[1]




# plot occurences of holiday related terms against the date,
# representing how often these terms appeared in tweets over the week

import matplotlib.pyplot as plt 
  
# line 3 points 
x3 = [19.0,20.0,21.0,22.0,23.0,24.0,25.0]
y3 = [t19, t20, t21, t22, t23, t24, t25] 
# plotting the line 2 points  
plt.plot(x3, y3, label = "Holiday Hashtags") 

# naming the x axis 
plt.xlabel('Day of November') 
# naming the y axis 
plt.ylabel('Thanksgiving related hashtags') 
# giving a title to my graph 
#plt.title('Sentiment Variation During Thanksgiving Week') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 

