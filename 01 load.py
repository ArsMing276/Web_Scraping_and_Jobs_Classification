# -*- coding: utf-8 -*-
"""
Load in data from data folder which contains several job categories
be sure to unzip part1 and part2 in to the data folder and create a folder called 
data_formatted  for further use

1. load data in recursively
2. remove html tags, remove punctuations, split data into training data and testing data
3. tokenization, remove stop words, and lemmatize the words
4. save training and testing data into data_formated folder.

"""
import glob
import re
from bs4 import BeautifulSoup as BS
from pandas import DataFrame
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

dirs = glob.glob("./data/*")

all_jobs = {}
for d in dirs:
    files = glob.glob(d + '/*')
    for file in files:
        op = open(file)
        lines = ' '.join(op.readlines())
        op.close()
        lines = BS(lines, "html.parser").getText(separator=u' ').lower()
        ## These words exist in almost all jobs and help nothing with prediction, so remove them
        lines = re.sub('(responsibilities|requirements?|qualifications?|job summary|about us|descriptions?)', '', lines)
        lines =  re.sub("[^a-zA-Z]", ' ', lines)
        lines = re.sub(' +', ' ', lines)
        cls = re.findall('data/(.+)/page', file)[0]
        ## create empty list when no matched key, otherwise, append the job description in to the corresponding value
        all_jobs.setdefault(cls, []).append(lines) 

##split into train_data and test_data
num = [len(x) for x in all_jobs.values()]
idx = [int(x*0.7) for x in num]
train_data = {}
test_data = {}


for i, key in enumerate(all_jobs.iterkeys()):
    ix = idx[i]
    train_data[key] = all_jobs[key][0:ix]
    test_data[key] = all_jobs[key][ix:]

##To transform the training data and test data into data frame, we need to 
#transform them to list at first
train_list = []
test_list = []
for key, vals in train_data.iteritems():
    for val in vals:
        train_list.append((key, val))

for key, vals in test_data.iteritems():
    for val in vals:
        test_list.append((key, val))

train_df = DataFrame(train_list, columns = ['category', 'description'])
test_df = DataFrame(test_list, columns = ['category', 'description'])

##tokenization, remove stop words and stem words
stops = set(stopwords.words("english"))           
wnl = WordNetLemmatizer()       

def word_process(lines):
    bags = word_tokenize(lines) 
    bags = [w for w in bags if not w in stops]  
    bags = [wnl.lemmatize(w) for w in bags]
    bags = ' '.join(bags)
    return bags

train_df['description'] = train_df['description'].map(word_process)
test_df['description'] = test_df['description'].map(word_process)

#dump to pickle file for further use
train_df.to_pickle('./data_formated/train_df')  
test_df.to_pickle('./data_formated/test_df')  