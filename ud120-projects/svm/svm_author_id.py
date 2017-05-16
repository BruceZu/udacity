#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
import numpy as np
from sklearn.svm import SVC
clf = SVC(kernel = "rbf", C = 10000.0 )

# slice the training dataset down to 1% of its original size,
# tossing out 99% of the training data
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
# training time: 177.66 s -- Much slower than Naive Bayes

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
# prediction time: 18.87 s

# find the prediction of a specific element:
answer10 = pred[10]
answer26 = pred[26]
answer50 = pred[50]

print "10th element:", answer10
print "26th element:", answer26
print "50th element:", answer50

# get accuracy score:
from sklearn.metrics import accuracy_score
print "accuracy score:", accuracy_score(pred, labels_test)

# how many are predicted to be in the “Chris” (1) class?
print 'Chris:', np.count_nonzero(pred == 1)
# method 2:
print 'Chris:', (pred == 1).sum()



#########################################################

