def mannycount():
    count = 0
    for x in range(3,118):
        print x
        if x%15==0:
            count+=15
        elif x%5==0:
            count+=5
        elif x%3==0:
            count+=3
        else:
            count+=1
    print count
    
            
mannycount()
    
    
    