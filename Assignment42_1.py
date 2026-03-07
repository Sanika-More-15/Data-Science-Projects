#cal mean of x 
# mean of y 
# slop of m 
# intercept c

import numpy as np
import matplotlib .pyplot as plt
import pandas as pd

def regration():
    #data set load
    X=np.array([1,2,3,4,5])
    Y=np.array([3,4,2,4,5])
    print ("independent variable:",X)
    print ("dependent variable:",Y)
    

    #calculate mean of x & y
    Mean_X=np.mean(X)
    Mean_Y=np.mean(Y)

    print("Mean of X is :",Mean_X)
    print("Mean of Y is :",Mean_Y)
    

    #calculate m
    #formula is x-mean(x)*y-mean(y)/x-mean(x)**2

    m=np.sum((X-Mean_X)*(Y-Mean_Y))/ np.sum((X-Mean_X)**2)
    print("Slop of m is :",m)
    
    #calculate intercept c
    #formula c=mean(y)-m*mean(x)

    c=Mean_Y - m * Mean_X
    print("intercept of c is :",c)

    #calculate regration equation 
    #formula y=m(x)+c
             #y=0.4*6+2.4

    Result=Y=m*6+c
    print("Predicted Y for X=6:",Result) 

def main():
    regration()

if __name__=="__main__":
    main()    