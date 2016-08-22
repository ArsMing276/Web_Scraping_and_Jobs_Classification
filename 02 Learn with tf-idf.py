# -*- coding: utf-8 -*-
"""
Create Features from Bag of Words
We will use three algorithms -- Naive Bayes, random forest and svm 
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from three_MLs import train_algo

train_df = pd.read_pickle('./data_formated/train_df')
test_df = pd.read_pickle('./data_formated/test_df')

vectorizer = TfidfVectorizer(analyzer = "word", max_df = 0.9, \
                             max_features = 500) 
                             
train_features = vectorizer.fit_transform(train_df['description'])
train_features = train_features.toarray()

vocab = vectorizer.get_feature_names()
imp = vectorizer.idf_

##print top 10 most important word(feature)
topword = [(x,y) for x,y in zip(vocab, imp)]
topword.sort(key=lambda x: x[1], reverse=True)
topword[0:50]

test_features = vectorizer.transform(test_df['description'])
test_features = test_features.toarray()
##rarely used words ranked before frequently used words, since they help more to
##in separating our documents

train_algo(train_features, train_df['description'], test_features, test_df['description'])









