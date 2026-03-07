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

    #calculate regration equation  of predicted all y values
    #formula yp=m*x=c
    Yp=m*X+c
    print("predicted y values are :",Yp)
    
    #calculate Mean Square Error(mse)
    #formula is  mse=i/n summation(y-yp)**2

    #calculate (y-yp)
    M=Y-Yp
   # print("value of Y-Yp is:",M)

    #calculate (y-yp)**2
    E=(M)**2
   # print(E) 

    #calculate mean of E
    mse=np.mean(E)
    print("MSE is :",mse)
    
   #calculate R**2 score
   # formula is R=1-summationm (y-yp)**2/(y-ybar) 
   
   #calculate sse
   #formula is summation(y-yp)**2
    #SSE=np.sum((Y-Yp)**2)
    SSE=np.sum((M)**2)

   #calculate sst
   # formula is summation of (y-ymean)**2
    SST=np.sum((Y-Mean_Y)**2)
           
  #calculate R**2 score 
    R=1-SSE/SST
    print("R**2 Score is :",R)
   

def main():
    regration()

if __name__=="__main__":
    main()    