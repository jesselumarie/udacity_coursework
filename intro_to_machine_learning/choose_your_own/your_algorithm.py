#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import RandomForestClassifier

best_leaf_value=[0,0, []]
test_range = range(1,500)
test_iterations = range(0,1000)
average_score_sum = 0

for x in test_range:
    clf = RandomForestClassifier(criterion = "entropy",min_samples_leaf=4) #create the random forest classifier
    clf = clf.fit(features_train, labels_train) #train the classifier

    pred = clf.predict(features_test) #create an array of predictions

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)  #determine the accuracy of those predictions    
    average_score_sum+=acc
    if acc > best_leaf_value[1]:
        best_leaf_value[0]=x  #store the leaf value which yields the highest accuracy
        best_leaf_value[1]=acc   #store the new highest average accuracy
        best_leaf_value[2]= clf.get_params(deep = True)
        
average_score = average_score_sum/len(test_range)

print "High Score: ", best_leaf_value[1]
print "Average Score: ", average_score
print "Deets: ", best_leaf_value


'''
for x in test_range:
    print x
    average_score_sum = 0
    for t in test_iterations:
        clf = RandomForestClassifier(min_samples_leaf=x) #create the random forest classifier
        clf = clf.fit(features_train, labels_train) #train the classifier

        pred = clf.predict(features_test) #create an array of predictions

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(pred, labels_test)  #determine the accuracy of those predictions
        average_score_sum+=acc #sum the average


    if average_score_sum/len(test_iterations) > best_leaf_value[1]: #after the loop, check if average is more than current best
        best_leaf_value[0]=x  #store the leaf value which yields the highest accuracy
        best_leaf_value[1]=average_score_sum/len(test_iterations)   #store the new highest average accuracy
        best_leaf_value[2]= clf.get_params(deep = True)
    

        
print "Average Score: ", best_leaf_value[1]
print "Deets: ", best_leaf_value
'''    


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
