import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def marvellousadvertise(datapath):

    border="-"*40

    # Step 1: load dataset
    print(border)
    print("Step 1: load dataset from csv")
    print(border)

    df=pd.read_csv(datapath)

    print("Some records from dataset")
    print(df.head())


    # Step 2: clean dataset
    print(border)
    print("Step 2: clean dataset")
    print(border)

    df.dropna(inplace=True)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'],inplace=True)

    print("Total records:",df.shape[0])
    print("Total columns:",df.shape[1])


    # Step 3: separate independent and dependent variable
    print(border)
    print("Step 3: separate independent and dependent variable")
    print(border)

    X=df[['TV','radio','newspaper']]
    Y=df['sales']

    print("Shape of X:",X.shape)
    print("Shape of Y:",Y.shape)

    print("Input columns:",X.columns.tolist())
    print("Output column: sales")


    # Step 4: split dataset
    print(border)
    print("Step 4: split dataset for training and testing")
    print(border)

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

    print("X_train shape:",X_train.shape)
    print("X_test shape:",X_test.shape)
    print("Y_train shape:",Y_train.shape)
    print("Y_test shape:",Y_test.shape)


    # Step 5: create and train model
    print(border)
    print("Step 5: create and train Linear Regression model")
    print(border)

    model=LinearRegression()

    model.fit(X_train,Y_train)

    print("Model training completed")


    # Step 6: test model
    print(border)
    print("Step 6: test the model")
    print(border)

    Y_pred=model.predict(X_test)


    # Step 7: evaluate model
    print(border)
    print("Step 7: evaluate the model")
    print(border)

    MSE=mean_squared_error(Y_test,Y_pred)
    RMSE=np.sqrt(MSE)
    R2=r2_score(Y_test,Y_pred)

    print("Mean Squared Error:",MSE)
    print("Root Mean Squared Error:",RMSE)
    print("R2 Score:",R2)


    # Step 8: model coefficients
    print(border)
    print("Step 8: model coefficients")
    print(border)

    for column,value in zip(X.columns,model.coef_):
        print(column,":",value)

    print("Intercept:",model.intercept_)


    # Step 9: compare actual vs predicted
    print(border)
    print("Step 9: compare actual vs predicted")
    print(border)

    result=pd.DataFrame({
        "Actual Sales":Y_test.values,
        "Predicted Sales":Y_pred
    })

    print(result.head())


def main():

    border="-"*40
    print(border)
    print("Advertising Sales Predictor using Linear Regression")
    print(border)

    marvellousadvertise("Advertising.csv")


if __name__=="__main__":
    main()