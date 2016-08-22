# -*- coding: utf-8 -*-
"""
Two approaches to generate new feature space with word2vec -- Embedding Vectorizer
Averaging and clustering
"""

##Embedding Vectorizer Averaging
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from three_MLs import train_algo

model = Word2Vec.load("./data_formated/100featuresword2vecmodel")
train_df = pd.read_pickle('./data_formated/train_df')
test_df = pd.read_pickle('./data_formated/test_df')

def mapmodel(model):
    EmbedVec = []
    for idx, key in enumerate(model.index2word):
        val = model.syn0[idx,:]
        EmbedVec.append((key, val))
    return EmbedVec

EmbedVec = DataFrame(mapmodel(model), columns = ['word', 'features'])
EmbedVec = EmbedVec.set_index('word')

## approach Embedding Vector Averaging
def MeanEmbedVec(df, EmbedVec):
    featureVec = np.zeros((df.shape[0], EmbedVec['features'][0].shape[0]))
    for i, doc in enumerate(df['description']):
        words = doc.split()
        wordvecs = EmbedVec.loc[words]
        mean_feature = wordvecs['features'].mean(skipna=True)
        featureVec[i,:] = mean_feature
    return featureVec
MeanVec_train = MeanEmbedVec(train_df, EmbedVec)
MeanVec_test = MeanEmbedVec(test_df, EmbedVec)

## three models---------------------------------------------------------

train_algo(MeanVec_train, train_df['description'], MeanVec_test, test_df['description'])
##----------------------------------------------------------------------

## approach clustering:

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

word_centroid_map = list(zip( model.index2word, idx ))
word_centroid_map = DataFrame(word_centroid_map, columns = ['word','cluster'])

# For the first 10 clusters
for cluster in xrange(0,10):
    #
    # Print the cluster number  
    print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
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
    
ClusterVec_train = ClustersVec(train_df, word_centroid_map, num_clusters)
ClusterVec_test = ClustersVec(test_df, word_centroid_map, num_clusters)

## three models---------------------------------------------------------
## Approach 1: random forest
train_algo(ClusterVec_train, train_df['description'], ClusterVec_test, test_df['description'])

##----------------------------------------------------------------------




















