import pandas as pd
from sklearn.linear_model import LinearRegression

def LinearRegression1():

    data = [
        { 'Study Hours' : 1, 'Marks' : 50 },
        { 'Study Hours' : 2, 'Marks' : 55 },
        { 'Study Hours' : 3, 'Marks' : 60 },
        { 'Study Hours' : 4, 'Marks' : 65 },
        { 'Study Hours' : 5, 'Marks' : 70 }
    ]

    df= pd.DataFrame(data)

    X=df[['Study Hours']]

    Y=df['Marks']

    model=LinearRegression()

    model.fit(X,Y)

    print("Coefficient :",model.coef_[0])

    print("Intercept :",model.intercept_)

def main():
    LinearRegression1()

if __name__=="__main__":
    main()