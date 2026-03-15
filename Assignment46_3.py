import pandas as pd
from sklearn.linear_model import LinearRegression

def LinearRegression1():

    data = [
        { 'Study Hours' : 1,'Sleep Hours':7 ,'Marks' : 50 },
        { 'Study Hours' : 2,'Sleep Hours':6 , 'Marks' : 55 },
        { 'Study Hours' : 3,'Sleep Hours':7 , 'Marks' : 60 },
        { 'Study Hours' : 4,'Sleep Hours':6 , 'Marks' : 65 },
        { 'Study Hours' : 5,'Sleep Hours':8 , 'Marks' : 70 }
    ]

    df= pd.DataFrame(data)

    X=df[['Study Hours','Sleep Hours']]

    Y=df['Marks']

    model=LinearRegression()

    model.fit(X,Y)

    print("Coefficient for study hours:",model.coef_[0])
    print("Coefficient for sleep hours:",model.coef_[1])


    print("Intercept :",model.intercept_)


def main():
    LinearRegression1()

if __name__=="__main__":
    main()