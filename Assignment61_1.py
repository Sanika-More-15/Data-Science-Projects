import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = [
    [25, 500, 12, 1, 2],
    [30, 700, 24, 0, 1],
    [45, 1200, 6, 5, 8],
    [50, 1500, 5, 6, 10],
    [28, 600, 18, 1, 1],
    [35, 800, 30, 0, 0],
    [48, 1400, 4, 7, 9],
    [52, 1600, 3, 8, 12],
    [27, 550, 20, 0, 1],
    [42, 1300, 8, 4, 7]
]

y=[
    0, 0, 1, 1, 0,
    0, 1, 1, 0, 1
]

# 0 = Customer will stay
# 1 = Customer will leave

# Load dataset
data = pd.DataFrame(X, columns=['Age', 'Monthly charges', 'Tenure', 'Complaints', 'Support calls'])
print(data)

# Clean the dataset 
print("Missing values in the dataset:")
print(data.isnull().sum())

# Apply Standard Scaler
scaler = StandardScaler()
data[['Age', 'Monthly charges', 'Tenure', 'Complaints', 'Support calls']] = scaler.fit_transform(data[['Age', 'Monthly charges', 'Tenure', 'Complaints', 'Support calls']])
print("\n",data)

# Train FNN model
X = data.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

new_customer = [[46, 1450, 5, 6, 9]]
new_customer_scaled = scaler.transform(new_customer)
prediction = model.predict(new_customer_scaled)
if prediction[0] == 0:
    print("\nThe customer is likely to stay.")
else:
    print("\nThe customer is likely to leave.")

# Evaluate accuracy
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



