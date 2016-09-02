# -*- coding: utf-8 -*-
"""
Latent Dirichilet Allocation is yet another transformation from bag-of-words 
counts into a topic space of lower dimensionality. LDA is a probabilistic extension
of LSA (Latent Semantic Analytsis, also called multinomial PCA), so LDA’s topics 
can be interpreted as probability distributions over words. These distributions 
are, just like with LSA, inferred automatically from a training corpus.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from gensim import corpora, models
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_pickle('./data_formated/train_df')
test_df = pd.read_pickle('./data_formated/test_df')

##Process train data
train_doc = train_df['description'].str.split()
train_doc = train_doc.values.tolist()

#create a Gensim dictionary from the texts
train_dict = corpora.Dictionary(train_doc)

#remove extremes 
train_dict.filter_extremes(no_below=10, no_above=0.7)

#convert the dictionary to a bag of words corpus for reference
corpus = [train_dict.doc2bow(text) for text in train_doc]

## build LDA model, may take a long time
## We will run online LDA (Hoffman), which is an algorithm that takes a chunk of 
## documents, updates the LDA model, takes another chunk, updates the model etc. 
## Online LDA can be contrasted with batch LDA, which processes the whole corpus 
## (one full pass), then updates the model, then another pass, another update...

lda = models.ldamodel.LdaModel(corpus, id2word=train_dict, 
                               num_topics=300, update_every=1, 
                               chunksize=10000, passes=1)

lda_name = "./data_formated/ldamodel"
lda.save(lda_name)

#Process test data                  
test_doc = test_df['description'].str.split()
test_doc = test_doc.values.tolist()

#create a Gensim dictionary from the texts
test_dict = corpora.Dictionary(test_doc)

#remove extremes 
test_dict.filter_extremes(no_below=10, no_above=0.7)

#convert the dictionary to a bag of words corpus for reference
corpus_ts = [test_dict.doc2bow(text) for text in test_doc]



#-------------------------------------------------------------------------
# print the most contributing words for 20 randomly selected topics
lda.print_topics(20)


## Next, we will use get_document_topics to get the probabilities of the 300 topics
## that each document contains, will transform both training and test data to a
## new feature space

train_topics = lda.get_document_topics(corpus)
test_topics = lda.get_document_topics(corpus_ts)
train_topics = [dict(train_topics[x]) for x in range(train_df.shape[0])]
test_topics = [dict(test_topics[x]) for x in range(test_df.shape[0])]

train_features = DataFrame(train_topics).fillna(0)
test_features = DataFrame(test_topics).fillna(0)

## Use KNN for classification---------------------------------------------------------
neigh = KNeighborsClassifier(n_neighbors=15)
knn = neigh.fit(train_features, train_df['category']) 

result = knn.predict(test_features)
knn_err = np.mean(result != test_df['category'])

print "KNN test error is %.2f" %knn_err

## a very high test error, this turns out to be a bad idea
##----------------------------------------------------------------------


