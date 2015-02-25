#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import csv


sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label


### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


d_keys = data_dict["METTS MARK"].keys()
d_keys.insert(0, "name")

list_dict =[]

list_dict.append(d_keys)


for name in data_dict:
    new_entry = []
    new_entry.append(name)
    
    for value in data_dict[name]:
        new_entry.append(data_dict[name][value])
    
    list_dict.append(new_entry)


myfile = open("enron_data.csv", 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

for row in list_dict:
    wr.writerow(row)


    
