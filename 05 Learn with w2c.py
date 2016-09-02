# -*- coding: utf-8 -*-
"""
As we have seen in the last script, each word has been projected to a 500 dimensions 
vector. We have also left out words with document-frequency less than 5 times, those words
help little with prediction.

There are two ways to reduce dimension according to the word2vec result

1. Average of Vector Embedding: We average all the embedding vectors in a given 
   job decription(each job description has many words, each word corresponds to a vector)
   to finally get a 500 dimensions vector as our features for that job.
2. We cluster the words into some bags according to the embedding vectors. Then we count the 
   frequency of each bag that the words in a given job description fall into, use this frequency
   table as our final features for that job. 
   
We will also apply tf-idf transformation to the feature space of the second way
(feature space of the first way is not a count matrix thus tf-idf transformation is
not applicable)

"""


import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from three_MLs import train_algo
from sklearn.feature_extraction.text import TfidfTransformer

model = Word2Vec.load("./data_formated/word2vecmodel")
train_df = pd.read_pickle('./data_formated/train_df')
test_df = pd.read_pickle('./data_formated/test_df')

## reshape the model result into a data frame with two columns, the word and it's 
## corresponding embedding vector
def mapmodel(model):
    EmbedVec = []
    for idx, key in enumerate(model.index2word):
        val = model.syn0[idx,:]
        EmbedVec.append((key, val))
    return EmbedVec

EmbedVec = DataFrame(mapmodel(model), columns = ['word', 'features'])
EmbedVec = EmbedVec.set_index('word')

## Approach1: Embedding Vector Averaging
def MeanEmbedVec(df, EmbedVec):
    featureVec = np.zeros((df.shape[0], EmbedVec['features'][0].shape[0]))
    for i, doc in enumerate(df['description']):
        words = doc.split()
        wordvecs = EmbedVec.loc[words]
        mean_feature = wordvecs['features'].mean(skipna=True)
        featureVec[i,:] = mean_feature
    return featureVec

## new feature space from Embedding Vector Averaging

MeanVec_train = MeanEmbedVec(train_df, EmbedVec)
MeanVec_test = MeanEmbedVec(test_df, EmbedVec)

## three models---------------------------------------------------------

train_algo(MeanVec_train, train_df['category'], MeanVec_test, test_df['category'])
##test error is high, we will primary focus on the second way ---clustering

##----------------------------------------------------------------------

## Approach2: clustering:

# Set number of clusters to be 0.2 of the vocabulary size
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

## initialize kmeans algorithm
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

##create word centroid map to map each word to its corresponding centroid index
word_centroid_map = list(zip( model.index2word, idx ))
word_centroid_map = DataFrame(word_centroid_map, columns = ['word','cluster'])

#Find all words for the first 10 clusters, we can see the clusters make sense
## for example, cluster 1 is about sports, cluster 5 is about location(city), etc.
for cluster in xrange(0,10):  
    print "\nCluster %d" % cluster
    idx = np.where(word_centroid_map['cluster'] == cluster)[0]
    words = word_centroid_map['word'][idx]
    print words

word_centroid_map = word_centroid_map.set_index('word')

def ClustersVec(df, word_centroid_map, num_clusters):
    featureVec = np.zeros((df.shape[0], num_clusters))
    for i, doc in enumerate(df['description']):
        words = doc.split()
        wordvecs = word_centroid_map.loc[words]
        freq = wordvecs.dropna().astype('int')['cluster'].value_counts()
        featureVec[i,freq.index] = freq.values
    return featureVec

## new feature space from clustering:  
ClusterVec_train = ClustersVec(train_df, word_centroid_map, num_clusters)
ClusterVec_test = ClustersVec(test_df, word_centroid_map, num_clusters)


## apply tf-idf transformation
vectorizer = TfidfTransformer() 
                             
##transform training and testing data
train_features = vectorizer.fit_transform(ClusterVec_train)
train_features = train_features.toarray()

test_features = vectorizer.transform(ClusterVec_test)
test_features = test_features.toarray()

## three models---------------------------------------------------------
train_algo(train_features, train_df['category'], test_features, test_df['category'])

##----------------------------------------------------------------------

