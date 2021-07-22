#!/usr/bin/env python
# coding: utf-8

# In[26]:


#libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt


# In[27]:


#set rendom seed
np.random.seed(50)


# In[28]:


#add the data
corpus = pd.read_csv("train_covid2.csv")
corpus= corpus.drop(columns=['title', 'X1', 'X2'])


# In[29]:


#  data pre processing
data_clean = corpus

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)
data_clean = pd.DataFrame(data_clean.text.apply(round1))

data_clean['label'] = corpus['label']



#  Tokenization : In this each entry in the corpus will be broken into set of words
data_clean['text']= [word_tokenize(entry) for entry in data_clean['text']]
#  Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
#  WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(data_clean['text']):
  #  Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    #  Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    #  pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        #  Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    #  The final processed set of words for each iteration will be stored in 'text_final'
    data_clean.loc[index,'text_final'] = str(Final_words)
   


# In[30]:


#uncomment below if you want to take a look at the updated dataset
#data_clean


# In[31]:


data_clean2 = data_clean
data_clean2 = data_clean2.drop(columns='text')


# In[32]:


#Prepare Train and Test Data sets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data_clean2['text_final'],data_clean2['label'],test_size=0.2)


# In[33]:


#encoding
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# In[34]:


#Word Vectorization
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data_clean2['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[35]:


#un comment the line below if you want to see the vocabulary that it has learned from the corpus!
## print(Tfidf_vect.vocabulary_)


# In[46]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use confusion matrix function to get confusion matrix
print("\nConfusion Matrix of SVM Classifier:\n")
print(confusion_matrix(Test_Y,predictions_SVM ))

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score : ",accuracy_score(predictions_SVM, Test_Y)*100)


# In[23]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score: ",accuracy_score(predictions_NB, Test_Y)*100)

# Use confusion matrix function to get confusion matrix
print("\nConfusion Matrix of Multinomial NB Classifier:\n")
print(confusion_matrix(Test_Y,predictions_NB))


# In[45]:


#fit the training dataset on the logistic classifier
Logistic = LogisticRegression()
Logistic.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Logistic.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Logistic regression Accuracy Score: ",accuracy_score(predictions_NB, Test_Y)*100)

# Use confusion matrix function to get confusion matrix
print("\nConfusion Matrix of Logistic regression Classifier:\n")
print(confusion_matrix(Test_Y,predictions_NB))


# In[ ]:




