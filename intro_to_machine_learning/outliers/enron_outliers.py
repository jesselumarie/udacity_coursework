#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop( "TOTAL", 0 )  #remove the total outlier
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary,bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


#how to get the biggest datavalue
max_sal =0
for point in data:
    if point[1]>max_sal:
        max_sal = point[1]
    
print max_sal

#another way =  print data.max()
    

for key, value in data_dict.iteritems():
    if value["salary"]!='NaN' and value["bonus"]!='NaN':
        if value["salary"]>1000000 and value["bonus"]>4000000:
            print "Key = ", key
            print "Salary: ", value["salary"]
            print "Bonus: ", value["bonus"]
   
