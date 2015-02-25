#!/usr/bin/python

import pickle
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(42)


### the words (features) and authors (labels), already largely processed
words_file = "../text_learning/your_word_data.pkl" ### like the file you made in the last mini-project 
authors_file = "../text_learning/your_email_authors.pkl"  ### this too

word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )

### test_size is the percentage of events assigned to the test set (remainder go into training)
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test  = vectorizer.transform(features_test).toarray()
feature_names = vectorizer.get_feature_names()

### a classic way to overfit is to use a small number
### of data points and a large number of features
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(features_train,labels_train)

print dtree.score(features_test,labels_test)

important_feat = dtree.feature_importances_


for key, value in enumerate(important_feat):
    if value>.2:
        print value
        print key
        print feature_names[key]
      


#draw visualization
'''  #can't draw this one... it's a multi-dimensional representation!
plt.scatter(features_train, labels_train, color="b", label="train data")
plt.scatter(features_test, labels_test, color="r", label="test data")
plt.plot(features_test, dtree.predict(labels_test), color="black")
plt.legend(loc=2)
plt.xlabel("features")
plt.ylabel("labels")
'''