import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

Border="-"*40
################################################################################################
# Step 1 : Write a Python program to load the final student_performance_ml.csv using pandas.
################################################################################################
print(Border)
print("Step 1 : A Python program to load the final student_performance_ml.csv using pandas.")
print(Border)

Dataset="student_performance_ml.csv"
df=pd.read_csv(Dataset)
print("Dataset get's loaded successfully\n")
print("Initial entries from dataset :")
print(df.head())

################################################################################################
# Step 2 : Data Analysis (EDA)
################################################################################################
print(Border)
print("Step 2 : Data Analysis")
print(Border)

print("shape of dataset :",df.shape)
print("Column Names :",list(df.columns))

print("class Distribution (FinalResult)")
print(df["FinalResult"].value_counts())

print("Statistical report of dataset")
print(df.describe())

################################################################################################
# Step 3 : Decide independent and dependent variables.
################################################################################################
print(Border)
print("Step 3 : Decide Indipendent & Dependent variable")
print(Border)

feature_cols=[
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "AssignmentsCompleted",
    "SleepHours"
    ]

X=df[feature_cols]
Y=df["FinalResult"]

print("X shape :",X.shape)
print("Y shape :",Y.shape)

################################################################################################
# Step 4 : Visualization of dataset
################################################################################################
print(Border)
print("Step 4 : Visualisation of dataset")
print(Border)

plt.figure(figsize=(7,5))

sns.scatterplot(x='StudyHours',y='PreviousScore',hue='FinalResult',data=df)
plt.show()

################################################################################################
# Step 5 : Split the dataset for traning and testing
################################################################################################
print(Border)
print("Step 5 : Split the dataset for traning and testing")
print(Border)

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

print("Data splitting activity done")

print("X - Independent :",X.shape) #(30,5)
print("Y - Dependent :",Y.shape)   #(30,)

print("X_train :",X_train.shape) #(24,5)
print("X_test :",X_test.shape)   #(6,5)
print("Y_train :",Y_train.shape) #(24,)
print("Y_test :",Y_test.shape)   #(6,)

################################################################################################
# Step 6 : Build the model
################################################################################################
print(Border)
print("Step 6 : Build the model")
print(Border)

print("We are going to use DecisionTreeClassifier")
model=DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    random_state=42
)

print("Model successfully created :",model)

################################################################################################
# Step 7 : Train the model
################################################################################################
print(Border)
print("Step 7 : Train the model")
print(Border)

model.fit(X_train,Y_train)
print("Model traning completed")

################################################################################################
# Step 8 : Test the model
################################################################################################
print(Border)
print("Step 8 : Test / Evaluate the model")
print(Border)

Y_pred=model.predict(X_test)
print("Model Evaluation complete")

print(Y_pred.shape)

print("Expected answers :")
print(Y_test)

print("Predicted answers :")
print(Y_pred)

################################################################################################
# Step 9 : Evaluate the model performance
################################################################################################
print(Border)
print("Step 9 : Evaluate the model performance")
print(Border)

#Training accuracy
Train_pred=model.predict(X_train)
Train_accuracy=accuracy_score(Y_train,Train_pred)

#Testing accuracy
Test_pred=model.predict(X_test)
Test_accuracy=accuracy_score(Y_test,Test_pred)

print("Traning Accuracy :",Train_accuracy*100)
print("Testing Accuracy :",Test_accuracy*100)

#Checking Fitting
if Train_accuracy - Test_accuracy > 0.10:
    print("Model is Overfitting")
elif Test_accuracy >Train_accuracy:
    print("Model is Underfitting")
else:
    print("Model is Well Fitted")
###################################################################################################

################################################################################################
# Step 10 : Plot confusion matrix
################################################################################################
print(Border)
print("Step 10 : Plot confusion matrix")
print(Border)
cm=confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix :")
print(cm)

print("Classification Report")
print(classification_report(Y_test,Y_pred))

disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Confusion matrix of Student performance dataset")
plt.show()

################################################################################################
# Step 11 : Predict result for new student
################################################################################################
print(Border)
print("Step 11 : Predict result for new student")
print(Border)

# New Student data
new_student=pd.DataFrame(
    [[6, 85, 66, 7, 7]],
    columns=feature_cols
)

prediction=model.predict(new_student)

if prediction[0]==1:
    print("Student will Pass")
else:
    print("Student will Fail")

