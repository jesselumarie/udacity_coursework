#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
from pprint import PrettyPrinter
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi","bonus", "salary","from_poi_perc", "to_poi_perc", "shared_receipt_with_poi'"]

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


not_poi = []
is_poi = []
poi_list = []

'''
for ii in data_dict:  #clean up the data to exclude 'NaN's
   for yy in data_dict[ii]:
       if data_dict[ii][yy] == 'NaN':
           data_dict[ii][yy]=0
'''

data_dict_features = data_dict['METTS MARK'].keys() #find the names of the available features

from_poi_perc = []
to_poi_perc = []
shared_emails = []

for name in data_dict: #for data that is = to 0, set value to 0 instead of trying to divide by it
    if data_dict[name]['from_messages'] != 0:
        from_person_to_poi = float(data_dict[name]['from_this_person_to_poi'])/float(data_dict[name]['from_messages'])
        to_poi_perc.append(from_person_to_poi) #save ratio to array, 
        data_dict[name]["to_poi_perc"] = from_person_to_poi #add data to dictionary
    else:
        from_person_to_poi = 0
        to_poi_perc.append(0) #save ratio to array, 
        data_dict[name]["to_poi_perc"] = 0
        
    if data_dict[name]['to_messages'] != 0:
        from_poi_to_person = float(data_dict[name]['from_poi_to_this_person'])/float(data_dict[name]['to_messages'])
        from_poi_perc.append(from_poi_to_person)
        data_dict[name]["from_poi_perc"] = from_poi_to_person #add data to dictionary
        
        shared_emails.append(data_dict[name]['shared_receipt_with_poi'])
        data_dict[name]["from_poi_perc"] = shared_emails.append(data_dict[name]['shared_receipt_with_poi']/data_dict[name]['to_messages'])
    else:
        from_poi_to_person = 0 #if there are no messages to the person, set to zero
        from_poi_perc.append(0)
        shared_emails.append(0)
        data_dict[name]["from_poi_perc"] = 0

    #creating a label list
    if data_dict[name]['poi']==1: 
        is_poi.append([from_person_to_poi, from_poi_to_person])
        poi_list.append(data_dict[name]['poi'])
    else:
        not_poi.append([from_person_to_poi, from_poi_to_person])
        poi_list.append(data_dict[name]['poi'])

is_poi_to = []
is_poi_from = []
not_poi_to = []
not_poi_from  = []


bonus_vals = []
salary_vals = []
director_fees = []
poi_color  = []


for i, data in enumerate(poi_list):
    if data==True:
        poi_color.append("b")
        poi_list[i]=1
    else:
        poi_color.append("r")
        poi_list[i]=0        
        
for data in not_poi:
    not_poi_to.append(data[0])
    not_poi_from.append(data[1])
for data in is_poi:
    is_poi_to.append(data[0])
    is_poi_from.append(data[1])
    
 
        
for emp in data_dict:  #creating feature lists from dataset
    bonus_vals.append(data_dict[emp]['bonus'])
    salary_vals.append(data_dict[emp]['salary'])
    director_fees.append(data_dict[emp]['director_fees'])
    shared_emails.append(data_dict[emp]['shared_receipt_with_poi'])
   


#split data into test/training array
#from sklearn.cross_validation import train_test_split
#feature_train, feature_test, label_train, label_test = train_test_split(is_poi_to, is_poi_from)


##visualize

plt.scatter(is_poi_to, is_poi_from, color="r", label="is poi")
plt.scatter(not_poi_to, not_poi_from, color="b", label="not poi")

#plt.plot(to_poi_test, reg.predict(to_poi_test), color="green", linewidth=3)
plt.legend(loc=2)
plt.xlabel("to_poi")
plt.ylabel("from_poi")
#plt.show()


plt.scatter(bonus_vals, salary_vals, color = poi_color)
#plt.show()



### we suggest removing any outliers before proceeding further
print max(bonus_vals)  #find ouliers and print'm
print max(salary_vals)

print bonus_vals.index(97343619)

del poi_list[104]
bonus_vals.remove(97343619)
salary_vals.remove(26704229)

plt.scatter(bonus_vals, salary_vals, color = poi_color)
#plt.show()


### if you are creating any new features, you might want to do that here




### store to my_dataset for easy export below
my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### if you are creating new features, could also do that here






### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)



### machine learning goes here!
### please name your classifier clf for easy export below
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


features = []
feature_names =["bonus_vals", "salary_vals","from_poi_perc", "to_poi_perc", "shared_emails"]

for x, item in  enumerate(bonus_vals):
    features.append([bonus_vals[x], salary_vals[x], from_poi_perc[x], to_poi_perc[x], shared_emails[x]])

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

feature_train, feature_test, label_train, label_test = train_test_split(features, poi_list)

'''---------Classifier Time---------------'''

clf = DecisionTreeClassifier(random_state=0)     ### get rid of this line!  just here to keep code from crashing out-of-box
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test) #create an array of predictions

from helper_functions import average_stats

score, precision, recall =  average_stats(clf, feature_test, label_test, 1)



print '\n'+"Decision Tree"
print "Score:", score
print "Precision:", precision
print "Recall:", recall

print np.array(feature_names)
print clf.feature_importances_

print "pred", clf.predict(feature_test) #print out the predictions vs the actual list
print "actl", np.array(label_test)

f1score = cross_val_score(clf, feature_train, label_train,
            cv=2, scoring='f1')          
print "F1 Score:",f1score



'''------------------------------------------------'''

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(random_state=2)
clf = clf.fit(feature_train, label_train) #train the classifier
pred = clf.predict(feature_test) #create an array of predictions
score, precision, recall =  average_stats(clf, feature_test, label_test, 1)

print '\n'+"Random Forest"
print "Score:", score
print "Precision:", precision
print "Recall:", recall

print np.array(feature_names)
print clf.feature_importances_


print "pred", clf.predict(feature_test) #print out the predictions vs the actual list
print "actl", np.array(label_test)

f1score = cross_val_score(clf, feature_train, label_train,
            cv=2, scoring='f1')          
print "F1 Score:",f1score

'''------------------------------------------------'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(feature_train, label_train) #train the classifier
pred = clf.predict(feature_test) #create an array of predictions
score, precision, recall =  average_stats(clf, feature_test, label_test, 1)

print '\n'+"Naive Bayes"
print "Score:", score
print "Precision:", precision
print "Recall:", recall

print "pred", clf.predict(feature_test) #print out the predictions vs the actual list
print "actl", np.array(label_test)

f1score = cross_val_score(clf, feature_train, label_train,
            cv=2, scoring='f1')          
print "F1 Score:",f1score
'''------------------------------------------------'''

### dump your classifier, dataset and features_list so 
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )



