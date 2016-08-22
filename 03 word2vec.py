# -*- coding: utf-8 -*-
"""
Create new feature space with word2vec

"""

import glob
import re
from bs4 import BeautifulSoup as BS
from pandas import DataFrame
from nltk.tokenize import PunktSentenceTokenizer
from operator import add
import logging
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

#we should not use test data info in the word2vec, extract training data only
num = [len(x) for x in all_jobs.values()]
idx = [int(x*0.7) for x in num]
keys = all_jobs.keys()
w2c_data = []

for key in keys:
    ix = idx[keys.index(key)]
    w2c_data += reduce(add, all_jobs[key][0:ix])


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 100    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(w2c_data, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "./data_formated/100featuresword2vecmodel"
model.save(model_name) 


