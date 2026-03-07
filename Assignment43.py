import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing 
import numpy as np


Border="-"*60

print(Border)
print("Step 1 : Load the dataset")
print(Border)

dataset="PlayPredictor.csv"
df=pd.read_csv(dataset)
print("Dataset gets loaded successfully")
print("Initial entries from dataset :")
print(df.head())

#-------------------------------------------------

print(Border)
print("Step 2 : Data Analysis")
print(Border)

print("shape of dataset :",df.shape)
print("Columns names :",list(df.columns))

print("Class distribution")
print(df["Play"].value_counts())

print("Statistical report of dataset")
print(df.describe())

#---------------------------------------------------

print(Border)
print("Step 3 : Decide Independent and Dependent variable")
print(Border)

# X: Independent variables / features
# Y: Dependent variables / features

feature_cols = [
    "Whether",
    "Temperature"
]

X = df[feature_cols]
Y = df["Play"]

print("X Shape :",X.shape)
print("Y Shape :",Y.shape)

#----------------------------------------------------

print(Border)
print("Step 4 : Visualization of dataset")
print(Border)

plt.figure(figsize=(7,5))

sns.scatterplot(x="Whether",y="Temperature",hue="Play",data=df)
plt.show()

#-----------------------------------------------------

print(Border)
print("Step 5 : Split the dataset for traning and testing")
print(Border)

le = preprocessing.LabelEncoder()
le_play=preprocessing.LabelEncoder()

#encode all string columns in your dataframe 'df'
df['Whether']=le.fit_transform(df['Whether'])
df['Temperature']=le.fit_transform(df['Temperature'])

X = df[['Whether','Temperature']]
Y=df['Play'] = le_play.fit_transform(df['Play'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

print("Data splitting activity done")

print("X_train :",X_train.shape) #(27,2)
print("X_test :",X_test.shape)   #(3,2) 
print("Y_train :",Y_train.shape) #(27,) 
print("Y_test :",Y_test.shape)   #(3,) 

#-------------------------------------------------------

print(Border)
print("Step 6 : Build the model")
print(Border)

print("We are going to use KNearestNeighboursClassifier")
model=KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)

accuracy=accuracy_score(Y_test,Y_pred)
print("accuracy is :",accuracy*100)

#----------------------------------------------------------
  
print(Border)
print("Step 7 : Testing the model with new data")
print(Border)

print("Labels and their numbers:")
for index, label in enumerate(le.classes_):
    print(f"{index} : {label}")

test_input = [[2,0]]

prediction = model.predict(test_input)

result_label = le_play.inverse_transform(prediction)

print(f"Testing for input :{test_input}")
print(f"Prediction : should we play ? -> {result_label[0]}")

print("\nTesting Multiple Scenarios:")
scenarios = [
    [0, 1], # Overcast (0) and Hot (1)
    [1, 0], # Rainy (1) and Cool (0)
    [2, 2]  # Sunny (2) and Mild (2)
]

multi_preds = model.predict(scenarios)
multi_labels = le_play.inverse_transform(multi_preds)

for i in range(len(scenarios)):
    print(f"Input {scenarios[i]} -> Result : {multi_labels[i]}")


#---------------------------------------------------------------

def CheckAccuracy(df):
    print("-" * 40)
    print("Step: Calculating Accuracy for different K")
    print("-" * 40)

    X = df[['Whether', 'Temperature']]
    Y = df['Play']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    k_values = [1, 3, 5, 7]
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        
        model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(Y_test, Y_pred)
        
        print(f"Accuracy of KNN with K={k} is : {accuracy * 100:.2f}%")

CheckAccuracy(df)
