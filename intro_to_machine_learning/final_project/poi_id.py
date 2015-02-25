#!/usr/bin/python

import sys
import pickle
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

<<<<<<< HEAD
### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ['poi', 'deferral_payments', 'deferred_income']
=======
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
>>>>>>> origin/master

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

<<<<<<< HEAD
### we suggest removing any outliers before proceeding further

for ii in data_dict:  #clean up the data to exclude 'NaN's (need to do this to find the outlier)
   for yy in data_dict[ii]:
       if data_dict[ii][yy] == 'NaN':
           data_dict[ii][yy]=0

#data_dict_features = data_dict['METTS MARK'].keys() #find the names of the available features

#for feature in data_dict_features:
    #print max(data_dict, key=lambda x: data_dict[x][feature])  #find the max value of a nested dictionary


data_dict.pop('TOTAL') # remove the outlier
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


### if you are creating any new features, you might want to do that here

for name in data_dict: #for data that is = to 0, set value to 0 instead of trying to divide by it
    if data_dict[name]['from_messages'] != 0:
        data_dict[name]["to_poi_perc"] = float(data_dict[name]['from_this_person_to_poi'])/float(data_dict[name]['from_messages'])
    else:
        data_dict[name]["to_poi_perc"] = 0
        
    if data_dict[name]['to_messages'] != 0:
        data_dict[name]["from_poi_perc"] = float(data_dict[name]['from_poi_to_this_person'])/float(data_dict[name]['to_messages'])
        data_dict[name]["shared_poi_perc"] =float(data_dict[name]['shared_receipt_with_poi'])/float(data_dict[name]['to_messages'])
    else:
        data_dict[name]["from_poi_perc"] = 0
        data_dict[name]["shared_poi_perc"] = 0


### store to my_dataset for easy export below


my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)


### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
=======
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
>>>>>>> origin/master
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

<<<<<<< HEAD
### machine learning goes here!
### please name your classifier clf for easy export below

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

feature_train, feature_test, label_train, label_test = train_test_split(features, labels, random_state=0)

'''*Testing Params*'''
cv_runs = 4  #number of folds in the cross val 
random_clf = 0

"""---------Random Forest---------------"""
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=random_clf)
clf = clf.fit(feature_train, label_train) #train the classifier
pred = clf.predict(feature_test) #create an array of predictions

np.set_printoptions(linewidth=1000)  #allow the array to print in a bigger area

print ("Random Forest")

accuracy = cross_val_score(clf, feature_train, label_train, cv=cv_runs, scoring='accuracy')       
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))  

recall =cross_val_score(clf, feature_train, label_train, cv=cv_runs, scoring='recall')       
print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))  

precision = cross_val_score(clf, feature_train, label_train, cv=cv_runs, scoring='average_precision')       
print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))

f1scores = cross_val_score(clf, feature_train, label_train, cv=cv_runs, scoring='f1')       
print("F1 Score: %0.2f (+/- %0.2f)" % (f1scores.mean(), f1scores.std() * 2))   


pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

print "Udacity_Class"
from my_tester import my_test 
my_test()
### dump your classifier, dataset and features_list so 
### anyone can run/check your results

""" #script that determines which features yield the highest precision/recall
from helper_functions import powerset

test = my_test()
features_list.remove('poi')

ps = powerset(features_list)  #list of features

best_features =[]
best_sum= 0.0


for s in ps: #iterate like crazy
    features_list = s

    if len(s)!=0 and len(s)!=1:
        w_poi = features_list
        w_poi.insert(0,'poi')
        data = featureFormat(my_dataset, features_list)

        ### split into labels and features (this line assumes that the first
        ### feature in the array is the label, which is why "poi" must always
        ### be first in features_list
        labels, features = targetFeatureSplit(data)


        ### machine learning goes here!
        ### please name your classifier clf for easy export below

        from sklearn.cross_validation import train_test_split, cross_val_score
        from sklearn.metrics import confusion_matrix

        feature_train, feature_test, label_train, label_test = train_test_split(features, labels, random_state=0)

        random_clf = 0 #set testing params
        
        #---------Random Forest---------------  #put test algorithm here
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier
        
        clf = DecisionTreeClassifier(random_state=random_clf)    
        clf.fit(feature_train, label_train)
        pred = clf.predict(feature_test) #create an array of predictions
        
        
        
        
        pickle.dump(clf, open("my_classifier.pkl", "w") )
        pickle.dump(data_dict, open("my_dataset.pkl", "w") )
        pickle.dump(features_list, open("my_feature_list.pkl", "w"))
    
        cur_sum = sum(my_test())
        print cur_sum
    
        if cur_sum>best_sum:
            best_features = s
            best_sum = cur_sum
            
            
print best_features, best_sum"""

=======
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)
>>>>>>> origin/master

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)