# -*- coding: utf-8 -*-
"""
Feature Engineering Approach1: Bag of words plus TFIDF
Create Features from Bag of Words and use TF-IDF to re-weight the features. 

Adopted three models -- random forest, naive bayes and SVM to train our data


by Tf-idf, even though we re-weighted words to help with prediction, this still has two constraint
 1. we could only use part of the words due to the dimensionality constraint, in other word, some infomation was ignored.
 2. Even among the kept words, some are useless. For example, diligent, every employer likes this but not all 
    companies would explicitly list this as a requirement. Thus this word is neither too frequenct nor too
    infrequent. Even it was kept in our final word list, it helps nothing in the classification.
 3. We will try two additional feature engineering approaches, Latent Dirichilet Allocation and word2vec.
    Both of them can achieve dimensionality reduction without losing much information.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from three_MLs import train_algo

train_df = pd.read_pickle('./data_formated/train_df')
test_df = pd.read_pickle('./data_formated/test_df')

## we don't want very frequent words or very infrequent words. They help nothing with prediction.
vectorizer = TfidfVectorizer(analyzer = "word", max_df = 0.75, \
                             min_df = 0.01, max_features = 5000) 
                             

##transform training and testing data
train_features = vectorizer.fit_transform(train_df['description'])
train_features = train_features.toarray()

test_features = vectorizer.transform(test_df['description'])
test_features = test_features.toarray()

##print top 50 most important word(feature)
vocab = vectorizer.get_feature_names()
imp = vectorizer.idf_

topword = [(x,y) for x,y in zip(vocab, imp)]
topword.sort(key=lambda x: x[1], reverse=True)
topword[0:50]


## three models---------------------------------------------------------
train_algo(train_features, train_df['category'], test_features, test_df['category'])

##----------------------------------------------------------------------




