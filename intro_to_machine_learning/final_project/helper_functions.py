#this function returns all of the values for a particular key, either split into poi/non-poi or together
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def average_stats(clf, feature_test, label_test, runs):
    ave_score =0.0
    ave_prec =0.0
    ave_recall = 0.0
    i=0
    
    while i < runs:
        pred = clf.predict(feature_test)
        ave_score+=clf.score(feature_test, label_test)
        ave_prec+=precision_score(label_test, pred) 
        ave_recall+=recall_score(label_test,pred)
        i+=1
        
    return ave_score/runs, ave_prec/runs, ave_recall/runs
    
    

def powerset(seq):
    
    if len(seq)<=1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield[seq[0]]+item
            yield item