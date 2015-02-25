#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from sklearn.svm import SVC
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
clf = SVC(kernel="rbf", C=10000) #specifies a  kernel

#features_train = features_train[:len(features_train)/100] #make the training dataset 1% of the original size
#labels_train = labels_train[:len(labels_train)/100] 

#t0= time()
clf.fit(features_train, labels_train) 
#print "training time:", round(time()-t0, 3), "s"


pred = clf.predict(features_test)  #make predictions based on the classifier

num_1 = 0
for preds in pred:
    if preds == 1:
        num_1+=1

print num_1


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)  #determine the accuracy of those predictions
print acc
#########################################################


