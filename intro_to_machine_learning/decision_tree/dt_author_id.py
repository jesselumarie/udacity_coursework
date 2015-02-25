#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
from sklearn import tree
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = tree.DecisionTreeClassifier(min_samples_split = 40) #create classifier
clf = clf.fit(features_train, labels_train) #train the classifier
pred = clf.predict(features_test) #predict using a small number of sample splits

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)  #determine the accuracy of those predictions

print acc
#########################################################


