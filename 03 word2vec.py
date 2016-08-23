# -*- coding: utf-8 -*-
"""
A brief introduction of word2vec
Word2vec, published by Google in 2013, is a neural network implementation that 
learns distributed representations for words. Word2vec learns quickly relative to other models.

Word2Vec does not need labels in order to create meaningful representations. This is useful, 
since most data in the real world is unlabeled. If the network is given enough training data 
(tens of billions of words), it produces word vectors with intriguing characteristics. Words 
with similar meanings appear in clusters, and clusters are spaced such that some word relationships, 
such as analogies, can be reproduced using vector math. The famous example is that, with highly
trained word vectors, "king - man + woman = queen."

-------------------------------------------------------------------------------
As we can see in the description, word2vec helps cluster words with similar meanings, 
by mapping each word to a vector and use some measurements (e.g. cosine distance) to
determine two words similarity. This is useful, but in this project, we will use word2vec 
to reduce dimentionality, and engineer new features according to the result.

"""

import glob
import re
from bs4 import BeautifulSoup as BS
from pandas import DataFrame
from nltk.tokenize import PunktSentenceTokenizer
from operator import add
from gensim.models import word2vec


# for word2vec it's better not to remove stop words and lemmatize the words
dirs = glob.glob("./data/*")
all_jobs = {}
for d in dirs:
    files = glob.glob(d + '/*')
    for file in files:
        op = open(file)
        lines = ' '.join(op.readlines())
        op.close()
        lines = BS(lines, "html.parser").getText(separator=u' ')
        sents = PunktSentenceTokenizer().tokenize(lines)
        sents = [re.sub("[^a-zA-Z]"," ", x) for x in sents if x]
        sents = [re.sub(' +', ' ', x) for x in sents]
        words = [x.lower().split() for x in sents]
        cls = re.findall('data/(.+)/page', file)[0]
        all_jobs.setdefault(cls, []).append(words)

#use training data only
num = [len(x) for x in all_jobs.values()]
idx = [int(x*0.7) for x in num]
keys = all_jobs.keys()
w2c_data = []

for key in keys:
    ix = idx[keys.index(key)]
    w2c_data += reduce(add, all_jobs[key][0:ix])


## set values for parameters
num_features = 500    # Word vector dimensionality                                            
num_workers = 4       # Number of threads to run in parallel
min_word_count = 30   # Minimum word counts
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(w2c_data, workers=num_workers, min_count = min_word_count, \
            size=num_features, window = context, sample = downsampling)

model.init_sims(replace=True)

##save model to the data_formated folder
model_name = "./data_formated/word2vecmodel"
model.save(model_name) 


