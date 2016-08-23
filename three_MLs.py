# -*- coding: utf-8 -*-
"""
Three Machine Learning methods
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


## running with carefulness. sklearn run very slowly when training such a big data
## Be sure to run it when you have set up all parameters correctly.

def train_algo(train_features, train_target, test_features, test_target):
    ## Approach 1: random forest
    rf = RandomForestClassifier(n_estimators = 100, oob_score = True) 
    model = rf.fit(train_features, train_target)

    ##look at out of bag error
    oob_error = 1 - model.oob_score_
    
    print "out of bag error of random forest is %.2f" %oob_error
    
    result = model.predict(test_features)
    rf_err = np.mean(result != test_target) 
    
    print "random forest test error is %.2f" %rf_err
    
    ## Approach 2: SVM
    svc = SVC(kernel='linear')
    model1 = svc.fit(train_features, train_target)
    result1 = model1.predict(test_features)
    svm_err1 = np.mean(result1 != test_target) 
    
    print "SVM with linear kernel test error is %.2f" %svm_err1  
    
    rbf_svc = SVC(kernel='rbf')
    model2 = rbf_svc.fit(train_features, train_target)
    result2 = model2.predict(test_features)
    svm_err2 = np.mean(result2 != test_target) 
    
    print "SVM with rbf kernel test error is %.2f" %svm_err2 
    
    poly_svc = SVC(kernel='poly')
    model3 = poly_svc.fit(train_features, train_target)
    result3 = model3.predict(test_features)
    svm_err3 = np.mean(result3 != test_target) 
      
    print "SVM with polynomial kernel test error is %.2f" %svm_err3 
    
    # Approach 3: Naive Bayes
    nb = MultinomialNB()
    model4 = nb.fit(train_features, train_target)
    result2 = model4.predict(test_features)
    nb_err = np.mean(result2 != test_target) 
    
    print "Naive Bayes test error is %.2f" %nb_err
    
