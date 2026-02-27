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

Dataset="student_performance_ml_2.csv"
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

################################################################################################
# Step 12 : Use model.feature_importance_(provides a score for each input feature, indicating its relative importance in predicting the target variable.)
################################################################################################
print(Border)
print("Step 12 : Use model.feature_importance_")
print(Border)

#Access the feature imp.
importance=model.feature_importances_

#combine with feature names for better interpretation
feature_names=X_test.columns #list of feature names

feature_importance_df=pd.DataFrame({
    'features':feature_names, 
    'importance':importance
    }).sort_values(by='importance',ascending=False)

print(feature_importance_df)
#Most Contributive Features:closer to 1.0 =         Attendance         1.0
#Least Contributive Features:scores close to 0.0  = StudyHours         0.0
                                                #PreviousScore         0.0
                                         #AssignmentsCompleted         0.0
                                                   #SleepHours         0.0 

################################################################################################
# Step 13 : Remove the column SleepHours from the dataset
################################################################################################
print(Border)
print("Step 13 : Remove the column SleepHours from the dataset")
print(Border)

df_new=df.drop('SleepHours',axis=1)
print("SleepHours Column removed successfully..\n",df_new.head())

# New feature list
new_feature_cols = [
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "AssignmentsCompleted"
]

# Independent & dependent variables again
X_new = df_new[new_feature_cols]
Y_new = df_new["FinalResult"]

# Split again
X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(
    X_new, Y_new, test_size=0.2, random_state=42
)

# Create NEW model
model_new = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

# Train again
model_new.fit(X_train_new, Y_train_new)

print("Model retrained successfully")

# Calculate New accuracy
# Training accuracy
Train_pred_new = model_new.predict(X_train_new)
Train_accuracy_new = accuracy_score(Y_train_new, Train_pred_new)

# Testing accuracy
Test_pred_new = model_new.predict(X_test_new)
Test_accuracy_new = accuracy_score(Y_test_new, Test_pred_new)

print("\nNew Training Accuracy :", Train_accuracy_new * 100)
print("New Testing Accuracy :", Test_accuracy_new * 100)

#Compare with Previous accuracy

print("\nAccuracy Comparison")
print("--------------------------")

print("old Testing accuracy :",Test_accuracy*100)
print("new Tesing accuracy :",Test_accuracy_new*100)

if Test_accuracy_new > Test_accuracy:
    print("Accuracy improved after removing SleepHours")
elif Test_accuracy_new < Test_accuracy:
    print("Accuracy Reduced after removing SleepHour")
else:
    print("Accuracy Remains Same after removing SleepHours")

#Removing SleepHours did not change the accuracy, which means
#SleepHours is not important for deciding whether a student passes or fails.

################################################################################################
# Step 14 : Train model using only SleepHours and Attendance
################################################################################################
print(Border)
print("Step 14 : Train model using only SleepHours and Attendance")
print(Border)

two_features=["SleepHours","Attendance"]

X_two=df[two_features]
Y_two=df["FinalResult"]

X_train_two, X_test_two, Y_train_two, Y_test_two = train_test_split(
    X_two, Y_two, test_size=0.2, random_state=42
)

model_two=DecisionTreeClassifier(random_state=42)
model_two.fit(X_train_two, Y_train_two)
print("Model trained using only two features")

#Calculate Accuracy
Train_pred_two=model_two.predict(X_train_two)
Test_pred_two=model_two.predict(X_test_two)

Train_accuracy_two= accuracy_score(Y_train_two, Train_pred_two)
Test_accuracy_two= accuracy_score(Y_test_two, Test_pred_two)

print("Training Accuracy (2 features) :", Train_accuracy_two * 100)
print("Testing Accuracy (2 features) :", Test_accuracy_two * 100)

print("\nAccuracy Comparison")
print("-----------------------")

print("Full Feature Accuracy :", Test_accuracy * 100)
print("Two Feature Accuracy :", Test_accuracy_two * 100)

if Test_accuracy_two > Test_accuracy:
    print("Two-feature model performs better")

elif Test_accuracy_two < Test_accuracy:
    print("Full-feature model performs better")

else:
    print("Both models perform equally")

################################################################################################
# Step 15 : Predict result for new students.
################################################################################################
print(Border)
print("Step 15 : Predict result for new students.")
print(Border)

new_students=pd.DataFrame({
    "StudyHours": [5, 8, 2, 6, 3],
    "Attendance": [80, 92, 60, 83, 71],
    "PreviousScore": [65, 88, 40, 55, 98],
    "AssignmentsCompleted": [6, 9, 3, 7, 4],
    "SleepHours":[7, 6, 8, 5, 9]
})

print("New Students Data : ")
print(new_students)

#predict using trained model
predictions=model.predict(new_students)

#Add prediction column
new_students["PredictedResult"]=predictions

#Display Results Clearly
print("\nPrediction Results :")
print(new_students)

################################################################################################
# Step 16 : Manual Accuracy Calculation.
################################################################################################
print(Border)
print("Step 16 : Manual Accuracy Calculation.")
print(Border)

manual_accuracy=(Y_test==Y_pred).sum() / len(Y_test)

print("Manual Accuracy :", manual_accuracy*100)
print("Sklearn Accuracy :", Test_accuracy*100)

################################################################################################
# Step 17 : Misclassified Students.
################################################################################################
print(Border)
print("Step 17 : Misclassified Students.")
print(Border)

misclassified= Y_test != Y_pred

wrong_students=X_test[misclassified]
print("Misclassified Students :")
print(wrong_students)

#Count Misclassified students
count=misclassified.sum()
print("\nNo of misclassified Students :",count)

################################################################################################
# Step 18 : Compare accuracy using different random_state
################################################################################################
print(Border)
print("Step 18 : Compare accuracy using different random_state")
print(Border)

states=[0, 10, 42]

for rs in states:

    #split dataset
    X_train,X_test,Y_train,Y_test = train_test_split(
        X,Y, test_size=0.2, random_state=rs
    )

    #creste model
    model=DecisionTreeClassifier(random_state=rs)

    #train model
    model.fit(X_train,Y_train)

    #predict 
    Y_pred=model.predict(X_test)

    #accuracy
    acc=accuracy_score(Y_test,Y_pred)

    print(f"Random State={rs} -> Testing Accuracy = {acc*100:.2f}%")

################################################################################################
# Step 19 : Decision tree Visualization, Use: from sklearn.tree import plot_tree
################################################################################################
print(Border)
print("Step 19 : Decision tree Visualization, using plot_tree")
print(Border)

plt.figure(figsize=(6,7))

plot_tree(model,
        feature_names=feature_cols,
        class_names=["Fail", "Pass"],
        filled=True,
        rounded=True,
        fontsize=10
        )
plt.title("Decision Tree for Student Performance")
plt.show()
#The [Attendance] root node feature is selected because it gives the best separation of pass and fail students.

################################################################################################
# Step 20 : Create new column PerformanceIndex
################################################################################################
print(Border)
print("Step 20 : Create new column PerformanceIndex")
print(Border)

df["PerformanceIndex"]=(df["StudyHours"]*2) + df["Attendance"]

feature_cols=["StudyHours", "Attendance", "PreviousScore", "AssignmentsCompleted", "SleepHours", "PerformanceIndex"]

x=df[feature_cols]
y=df["FinalResult"]

model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(df.head())

################################################################################################
# Step 21 : Train model with : max_depth=None
################################################################################################
print(Border)
print("Step 21 : Train model with : max_depth=None")
print(Border)

X=df.drop(columns=["FinalResult"])
Y=df["FinalResult"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=None)
model.fit(X_train, Y_train)

print("Training Accuracy :", model.score(X_train, Y_train)*100)
print("Testing Accuracy :", model.score(X_test, Y_test)*100)


