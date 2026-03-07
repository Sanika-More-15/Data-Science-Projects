
#train regration model  
# predict salary for 6 yaear expirance
# plot regration line using matplotlib

import numpy as np
import matplotlib .pyplot as plt
from sklearn.linear_model import LinearRegression

def salary():
    #load data set
    x=np.array([1,2,3,4,5]).reshape(-1,1) #expiriance
    y=np.array([20000,25000,30000,35000,40000])#salary
    
    model=LinearRegression() #create

    model.fit(x,y) #train

    print("slop of m:", model.coef_) # calculate m
    print("intercept of c:", model.intercept_) #calculate c

    # predicted yp
    yp=model.predict(x)
    print("predicted yp is :",yp)

#predicted salory for 6 year expirance
    new=np.array([[6]])
    predi=model.predict(new)
    print("Predicted Salary for 6 year Expirance is :",predi)

#graph
    plt.scatter(x,y,color='red')
    plt.plot(x,yp,color='blue')

    plt.xlabel("Expiriance")
    plt.ylabel("Salary")
    plt.title(" linear regration model")

    plt.show()

def main():
    salary()
 
if __name__=="__main__":
    main()    
