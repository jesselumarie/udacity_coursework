#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import os

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

count_salary=0
count_pois = 0
count_payments =0 



for name in enron_data:
    if enron_data[name]['total_payments']=='NaN':
        count_payments+=1
        
        
print count_payments
print float(count_payments+10)/float(len(enron_data)+10)