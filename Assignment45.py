#using  DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def MarvellousWinePredictor(DataPath):

    Border="-"*60

    #------------------------------------------------------------
    # Step 1 : Load dataset
    #------------------------------------------------------------
    print(Border)
    print("Step 1 : Load dataset")
    print(Border)

    df=pd.read_csv(DataPath)

    print("Few records from dataset :")
    print(df.head())


    #------------------------------------------------------------
    # Step 2 : Remove unwanted columns
    #------------------------------------------------------------
    print(Border)
    print("Step 2 : Remove unwanted columns")
    print(Border)

    print("Shape before removal :",df.shape)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'],inplace=True)

    print("Shape after removal :",df.shape)


    #------------------------------------------------------------
    # Step 3 : Check missing values
    #------------------------------------------------------------
    print(Border)
    print("Step 3 : Check missing values")
    print(Border)

    print(df.isnull().sum())


    #------------------------------------------------------------
    # Step 4 : Statistical summary
    #------------------------------------------------------------
    print(Border)
    print("Step 4 : Statistical summary")
    print(Border)

    print(df.describe())


    #------------------------------------------------------------
    # Step 5 : Correlation
    #------------------------------------------------------------
    print(Border)
    print("Step 5 : Correlation")
    print(Border)

    print(df.corr())


    #------------------------------------------------------------
    # Step 6 : Independent & Dependent variables
    #------------------------------------------------------------
    print(Border)
    print("Step 6 : Split independent and dependent variables")
    print(Border)

    X=df.drop('Class',axis=1)
    Y=df['Class']

    print("Shape of X :",X.shape)
    print("Shape of Y :",Y.shape)


    #------------------------------------------------------------
    # Step 7 : Train test split
    #------------------------------------------------------------
    print(Border)
    print("Step 7 : Train test split")
    print(Border)

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


    #------------------------------------------------------------
    # Step 8 : Create & Train Model
    #------------------------------------------------------------
    print(Border)
    print("Step 8 : Create & Train Model")
    print(Border)

    model=DecisionTreeClassifier()

    model.fit(X_train,Y_train)


    #------------------------------------------------------------
    # Step 9 : Test Model
    #------------------------------------------------------------
    print(Border)
    print("Step 9 : Test Model")
    print(Border)

    Y_pred=model.predict(X_test)


    #------------------------------------------------------------
    # Step 10 : Evaluate Model
    #------------------------------------------------------------
    print(Border)
    print("Step 10 : Evaluate Model")
    print(Border)

    accuracy=accuracy_score(Y_test,Y_pred)

    print("Accuracy :",accuracy)

    print("Confusion Matrix :")
    print(confusion_matrix(Y_test,Y_pred))

    print("Classification Report :")
    print(classification_report(Y_test,Y_pred))

    #------------------------------------------------------------
    # Step 11 : Compare actual vs predicted
    #------------------------------------------------------------
    print(Border)
    print("Step 11 : Compare actual vs predicted")
    print(Border)

    Result=pd.DataFrame({
        "Actual Class":Y_test.values,
        "Predicted Class":Y_pred
    })

    print(Result.head())


def main():
    MarvellousWinePredictor("WinePredictor.csv")


if __name__=="__main__":
    main()