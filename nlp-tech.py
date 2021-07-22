#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Here we are going to test wordcloud,word sentiment and topic modeling


# In[2]:


#surpress warnings
import warnings
warnings.simplefilter("ignore")
import csv


# In[3]:


#Importing libraries:
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
import nltk
from sklearn import metrics
import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn


# In[4]:


#loading the data in a dataframe
data = pd.read_csv("final_data_corpus.csv", sep=';')
data


# In[5]:


#data cleaning
#lower_case the data
data_clean = data
def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)
data_clean = pd.DataFrame(data_clean.texts.apply(round1))

data_clean['label'] = data['label']

#lets take a look on our data !💗
data_clean


# In[6]:


#  data pre processing

 #start tokenizing
    
data_clean2 = data_clean['texts']
fake = nltk.word_tokenize(data_clean2.iloc[0])
real = nltk.word_tokenize(data_clean2.iloc[1])

  #stemming 

stemming = PorterStemmer()
fake = [stemming.stem(word) for word in fake]
real = [stemming.stem(word) for word in real]

  #stopwords
    
#now we have noticed that there some common few words between fake & real, so we are adding them to stopword list#stopwords = nltk.corpus.stopwords.words('english')

stops = stopwords.words('english')

fake = [w for w in fake if not w in stops]
real = [w for w in real if not w in stops]
  
  #re-join words
fake = (" ".join(fake))
real = (" ".join(real))

data_clean2[0] = fake
data_clean2[1] = real
data_clean2


# In[7]:


# wordcloud for fake news

wordcloud = WordCloud(
                          background_color='white',
                          
                          max_words=5000,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(data_clean.texts[0]))
plt.imshow(wordcloud)
plt.rcParams["figure.figsize"] = (16,17)
plt.axis('off')
plt.title("Title")


# In[8]:


#wordcloud for real news
wordcloud = WordCloud(
                          background_color='white',
                          
                          max_words=5000,
                          max_font_size=50, 
                          random_state=45
                         ).generate(str(data_clean.texts[1]))
plt.imshow(wordcloud)
plt.rcParams["figure.figsize"] = (16,17)
plt.axis('off')
plt.title("Title")


#  Now obviously wordcloud didn't say much about the diffrences but u can tell that the fake news talks more about vaccine/
# cure/ the lockdown  while real news intrests in patient/health/goverenment
# yet we have to try other techniques to find out more
# 

# In[9]:



#word sentiment(polarity,subjectivity) & plot it [fake]
from textblob import TextBlob
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data_clean['polarity'] = data_clean.texts.apply(pol)
data_clean['subjectivity'] = data_clean.texts.apply(sub)


plt.rcParams['figure.figsize'] = [10, 8]
for index, label in enumerate(data_clean.index):
    x = data_clean.polarity.iloc[label]
    y = data_clean.subjectivity.iloc[label]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['label'][index], fontsize=10)
    plt.xlim(-0.01, 0.35) 
    plt.ylim(-1,4)
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()


# In[10]:


# Which makes sense, both fake and real news are written in formal language, thats why both closer to [facts],
#and fake news tend to give more positive misinformation about the virus so that people don't take it seriously, there for it's #closer to positive.


# In[11]:


#Now lets start with Topic modeling attempt 1
#Latent Dirichlet Allocation (LDA) is a technique specially for topic modeling text data which we are using for this model


# In[12]:


#we will make a DTM form of our data

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.texts)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
#remove the obvious awkword words
data_dtm = data_dtm.drop(columns=['abid','abl','zonal','zoonot','york','wwwswrindianrailwaysgovin','abov','accord','yearsaft'])
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_clean.to_pickle("dtm_stop.pkl")


# In[13]:


# Import the necessary modules for LDA with gensim
from gensim import matutils, models
import scipy.sparse

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[14]:


# One of the required inputs is a term-document matrix
tdm = data_dtm.transpose()
tdm.head()


# In[15]:


# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)


# In[16]:


# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
cv = pickle.load(open("cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())


# In[27]:


# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)
lda.print_topics()


# In[28]:


#topic modeling attempt 2
# Let's create a function to pull out nouns from a string of text
from nltk import word_tokenize, pos_tag

def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)


# In[30]:


data_nouns = pd.DataFrame(data_clean.texts.apply(nouns))
data_nouns


# In[20]:


# Create a new document-term matrix using only nouns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['coronaviru', 'ha', 'thi', 'indian', 'peopl','wwwswrindianrailwaysgovin','abid','yearsaft','york','zoonot','countri'
                   'got',  'think', 'said','wa','india','group','report']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.texts)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index
data_dtmn


# In[21]:


# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())


# In[22]:


# Let's start with 2 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=2, id2word=id2wordn, passes=10)
ldan.print_topics()


# In[31]:


#topic modeling attempt 3 nouns and adjectives
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)
data_nouns_adj = pd.DataFrame(data_clean.texts.apply(nouns_adj))
data_nouns_adj


# In[33]:


# Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.texts)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index
data_dtmna


# In[34]:


# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())


# In[35]:


#with 2 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=2, id2word=id2wordna, passes=10)
ldana.print_topics()


# In[36]:


# By looking at the topics ,, it seems that the first topic is about 'infections/ hospitals/symptoms'
# the second topic is more about 'vaccine/ immmuny system/care'


# In[37]:


# Let's take a look at which topics each transcript contains (topic , lebel[0->fake,1->real])
corpus_transformed = ldana[corpusna]
list(zip([a for [(a,b)] in corpus_transformed], data_dtmna.index))


# In[ ]:




