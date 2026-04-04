import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------

df = pd.read_csv("bank-full.csv", sep=";")
df.columns = df.columns.str.replace('"', '')

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

#---------------------------------------------------------------------------------------
# Step 2 : Preprocessing
#---------------------------------------------------------------------------------------

# Label Encoding (only binary columns)
le = LabelEncoder()
binary_cols = ['default', 'housing', 'loan', 'y']

for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-Hot Encoding (multi-category columns)
df = pd.get_dummies(df, columns=[
    'job', 'marital', 'education', 'contact', 'month', 'poutcome'
], drop_first=True)

print("\nAfter Encoding Shape:", df.shape)

#---------------------------------------------------------------------------------------
# Step 3 : Split Features & Target
#---------------------------------------------------------------------------------------

X = df.drop("y", axis=1)
Y = df["y"]

#---------------------------------------------------------------------------------------
# Step 4 : Scaling
#---------------------------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#---------------------------------------------------------------------------------------
# Step 5 : Train-Test Split
#---------------------------------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

#---------------------------------------------------------------------------------------
# Step 6 : Decision Tree
#---------------------------------------------------------------------------------------

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

y_pred_dt = dt.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(Y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred_dt))
print("Classification Report:\n", classification_report(Y_test, y_pred_dt, zero_division=0))

y_prob_dt = dt.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(Y_test, y_prob_dt))

#---------------------------------------------------------------------------------------
# Step 7 : KNN
#---------------------------------------------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

y_pred_knn = knn.predict(X_test)

print("\n--- KNN ---")
print("Accuracy:", accuracy_score(Y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred_knn))
print("Classification Report:\n", classification_report(Y_test, y_pred_knn, zero_division=0))

y_prob_knn = knn.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(Y_test, y_prob_knn))

#---------------------------------------------------------------------------------------
# Step 8 : Logistic Regression
#---------------------------------------------------------------------------------------

lr = LogisticRegression(max_iter=5000)
lr.fit(X_train, Y_train)

y_pred_lr = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(Y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred_lr))
print("Classification Report:\n", classification_report(Y_test, y_pred_lr, zero_division=0))

y_prob_lr = lr.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(Y_test, y_prob_lr))

#---------------------------------------------------------------------------------------
# Step 9 : Visualization 
#---------------------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
