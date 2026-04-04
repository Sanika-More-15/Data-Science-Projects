import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------

data =[
    {'Pregnancies':6, 'Glucose':148, 'BloodPressure':72, 'SkinThickness':35, 'Insulin':0, 'BMI':33.6, 'DiabetesPedigreeFunction':0.627, 'Age':50, 'Outcome':1},
    {'Pregnancies':1, 'Glucose':85, 'BloodPressure':66, 'SkinThickness':29, 'Insulin':0, 'BMI':26.6, 'DiabetesPedigreeFunction':0.351, 'Age':31, 'Outcome':0},
    {'Pregnancies':8, 'Glucose':183, 'BloodPressure':64, 'SkinThickness':0, 'Insulin':0, 'BMI':23.3, 'DiabetesPedigreeFunction':0.672, 'Age':32, 'Outcome':1},
    {'Pregnancies':1, 'Glucose':89, 'BloodPressure':66, 'SkinThickness':23, 'Insulin':94, 'BMI':28.1, 'DiabetesPedigreeFunction':0.167, 'Age':21, 'Outcome':0},
    {'Pregnancies':0, 'Glucose':137, 'BloodPressure':40, 'SkinThickness':35, 'Insulin':168, 'BMI':43.1, 'DiabetesPedigreeFunction':2.288, 'Age':33, 'Outcome':1},
    {'Pregnancies':5, 'Glucose':116, 'BloodPressure':74, 'SkinThickness':0, 'Insulin':0, 'BMI':25.6, 'DiabetesPedigreeFunction':0.201, 'Age':30, 'Outcome':0},
    {'Pregnancies':3, 'Glucose':78, 'BloodPressure':50, 'SkinThickness':32, 'Insulin':88, 'BMI':31, 'DiabetesPedigreeFunction':0.248, 'Age':26, 'Outcome':1},
    {'Pregnancies':10, 'Glucose':115, 'BloodPressure':0, 'SkinThickness':0, 'Insulin':0, 'BMI':35.3, 'DiabetesPedigreeFunction':0.134, 'Age':29, 'Outcome':0},
    {'Pregnancies':2, 'Glucose':197, 'BloodPressure':70, 'SkinThickness':45, 'Insulin':543, 'BMI':30.5, 'DiabetesPedigreeFunction':0.158, 'Age':53, 'Outcome':1},
    {'Pregnancies':8, 'Glucose':125, 'BloodPressure':96, 'SkinThickness':0, 'Insulin':0, 'BMI':0, 'DiabetesPedigreeFunction':0.232, 'Age':54, 'Outcome':1},
    {'Pregnancies':4, 'Glucose':110, 'BloodPressure':92, 'SkinThickness':0, 'Insulin':0, 'BMI':37.6, 'DiabetesPedigreeFunction':0.191, 'Age':30, 'Outcome':0},
    {'Pregnancies':10, 'Glucose':168, 'BloodPressure':74, 'SkinThickness':0, 'Insulin':0, 'BMI':38, 'DiabetesPedigreeFunction':0.537, 'Age':34, 'Outcome':1},
    {'Pregnancies':10, 'Glucose':139, 'BloodPressure':80, 'SkinThickness':0, 'Insulin':0, 'BMI':27.1, 'DiabetesPedigreeFunction':1.441, 'Age':57, 'Outcome':0}
]

df=pd.DataFrame(data)

X = df.drop("Outcome",axis=1)
Y = df["Outcome"]

print("Few records from dataset :")
print(df.head())

#---------------------------------------------------------------------------------------
# Step 2 : EDA
#---------------------------------------------------------------------------------------

print("\nShape before removal :",df.shape)

print("\n",df.isnull().sum())

print("\n",df.describe())

#---------------------------------------------------------------------------------------
# Step 3 : Visualization (Use graph like hist.boxplot, or pairplot to identify patterns or outliers)
#---------------------------------------------------------------------------------------

plt.hist(df['Outcome'], bins=2)
plt.title("Distribution of Outcomes")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()

plt.boxplot(df['Outcome'])
plt.title("Boxplot of Outcome")
plt.show()

sns.pairplot(df, hue='Outcome')
plt.show()

#---------------------------------------------------------------------------------------
# Step 4 : Data Preprocessing (Handle missing values, Encode categorical variables, Scale features if necessary)
#---------------------------------------------------------------------------------------

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols:
    df[col] = df[col].replace(0, df[col].median())

scalar = StandardScaler()
X_Scaled = scalar.fit_transform(X)

x_train, X_test, Y_train, Y_test = train_test_split(X_Scaled,Y, test_size=0.3, random_state=42)

#----------------------------------------------------------------------
dt = DecisionTreeClassifier()
dt.fit(x_train, Y_train)
y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Accuracy :",accuracy_score(Y_test, y_pred_dt))

cn_dt = confusion_matrix(Y_test, y_pred_dt)
print("\nConfusion Matrix for Decision Tree :\n", cn_dt)

print("\nClassification Report for dt :")
print(classification_report(Y_test, y_pred_dt, zero_division=0))

#----------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, Y_train)
y_pred_knn = knn.predict(X_test)

print("\nKNN Accuracy :",accuracy_score(Y_test,y_pred_knn))

cn_knn = confusion_matrix(Y_test, y_pred_knn)
print("\nConfusion Matrix for KNN :\n", cn_knn)

print("\nClassification Report for knn :")
print(classification_report(Y_test, y_pred_knn, zero_division=0))

#----------------------------------------------------------------------
Lr =LogisticRegression()
Lr.fit(x_train, Y_train)
y_pred_lr = Lr.predict(X_test)

print("\nLogistic Regression Accuracy :",accuracy_score(Y_test,y_pred_lr))

cn_lr = confusion_matrix(Y_test, y_pred_lr)
print("\nConfusion Matrix for Logistic Regression :\n", cn_lr)

print("\nClassification Report :")
print(classification_report(Y_test, y_pred_lr, zero_division=0))

#---------------------------------------------------------------------------------------
# Step 5 : Final Output
#---------------------------------------------------------------------------------------

#Predict whether a patient is dibetic based in test data
y_pred_lr = Lr.predict(X_test)

#Display predictions on screen and save thein in csv file.
predictions = pd.DataFrame({
    'Actual': Y_test.values,
    'Predicted': y_pred_lr
})

print("\nPrediction Results:")
print(predictions)

predictions.to_csv("diabetes_predictions.csv", index=False)
print("\nPredictions saved to diabetes_predictions.csv")