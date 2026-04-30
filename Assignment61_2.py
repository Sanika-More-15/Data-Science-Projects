import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = [
    [25000, 600, 200000, 10000, 0],
    [40000, 700, 300000, 8000, 1],
    [60000, 750, 500000, 12000, 1],
    [20000, 550, 150000, 15000, 0],
    [80000, 800, 700000, 10000, 1],
    [35000, 650, 250000, 9000,  1],
    [18000, 500, 100000, 12000, 0],
    [90000, 850, 800000, 15000, 1],
    [30000, 580, 200000, 14000, 0],
    [70000, 780, 600000, 10000, 1]
]

y=[
    0, 1, 1, 0, 0,
    1, 0, 1, 0, 1
]

# 0 = Loan Rejected
# 1 = Loan Approved

# Load dataset
data = pd.DataFrame(X, columns=['Income', 'Credit Score', 'Loan Amount', 'Existing EMI', 'Employment Status'])
print(data)

# Employment Status: 0 = Not Stable, 1 = Stable

# Clean the dataset 
print("Missing values in the dataset:")
print(data.isnull().sum())

# Apply Standard Scaler
scaler = StandardScaler()
data[['Income', 'Credit Score', 'Loan Amount', 'Existing EMI', 'Employment Status']] = scaler.fit_transform(data[['Income', 'Credit Score', 'Loan Amount', 'Existing EMI', 'Employment Status']])
print("\n",data)

# Train FNN model
X = data.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

new_data = pd.DataFrame([[55000, 720, 400000, 10000, 1]], columns=['Income', 'Credit Score', 'Loan Amount', 'Existing EMI', 'Employment Status'])
new_applicant_scaled = scaler.transform(new_data)
prediction = model.predict(new_applicant_scaled)

if prediction[0] == 0:
    print("\nThe loan application is likely to be rejected.")
else:
    print("\nThe loan application is likely to be approved.")

# Evaluate accuracy
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



