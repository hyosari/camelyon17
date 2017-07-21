import matplotlib.pyplot as plt
import numpy as np 

def subplot_show(imagelist,row,col,name):
    """
        plot images which are in the imagelist 
        with row , col matrix form 

    """
    plt.figure(name)

    if len(imagelist) != row*col:
        
        print "rowxcol != len(imagelist)" 
    else:
        for i in range(len(imagelist)):
            plt.subplot(row,col,i+1)
            plt.imshow(imagelist[i])
            plt.axis('off') 
    
    
    return 0 
