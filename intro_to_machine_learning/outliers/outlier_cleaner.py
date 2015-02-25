#!/usr/bin/python
def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []
    
    ### your code goes here
    
    
    
    for x, item in enumerate(predictions):
        error = abs((item-net_worths[x])/net_worths[x])
        cleaned_data.append((ages[x], net_worths[x],error)) #add items as tuple
        
    
    cleaned_data.sort(key=lambda x: x[2])  #sort data by error
    
    
    cleaned_data = cleaned_data[:81]
    
    return cleaned_data

