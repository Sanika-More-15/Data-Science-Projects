import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier

#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0 # fake news
true_df["label"] = 1 # true news

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True) # shuffle the dataset

print("\n----------------------------------------------------------------")
print(df.head())
print(df.shape)
print(df.isnull().sum())

df["content"] = df["title"] + " " + df["text"]
df = df.dropna()

#---------------------------------------------------------------------------------------
# Step 2 : Use TF-IDF to convert text data into numerical features
#---------------------------------------------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=5000
)

x = tfidf.fit_transform(df["content"])
y = df["label"]

#---------------------------------------------------------------------------------------
# Step 3 : Train test split the model
#---------------------------------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
    )

#---------------------------------------------------------------------------------------
# Step 4 : Train model ( decision tree and logistic regression)
#---------------------------------------------------------------------------------------
#print("\n--- Logistic Regression ---")
lr = LogisticRegression()
lr.fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)

#print("\n--- Decision Tree ---")
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
y_pred_dt = dt.predict(X_test)

print("confusion matrix - Logistic Regression:")
print(confusion_matrix(Y_test, y_pred_lr))

print("confusion matrix - Decision Tree:")
print(confusion_matrix(Y_test, y_pred_dt))


#---------------------------------------------------------------------------------------
# Step 5 : Voting (hard and soft)
#---------------------------------------------------------------------------------------

# Hard vioting (majority voting)
hard_voting = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt)],
    voting='hard'
)

# Soft voting (average probabilities)
soft_voting = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt)],
    voting='soft'
)

# Train both
hard_voting.fit(X_train, Y_train)
soft_voting.fit(X_train, Y_train)

#---------------------------------------------------------------------------------------
# Step 6 : Model Evaluation
#---------------------------------------------------------------------------------------
y_pred_hard = hard_voting.predict(X_test)
print("Accuracy hard voting:", accuracy_score(Y_test, y_pred_hard))

y_pred_soft = soft_voting.predict(X_test)
print("Accuracy soft voting:", accuracy_score(Y_test, y_pred_soft))

print("\nConfusion Matrix - Hard Voting")
print(confusion_matrix(Y_test, y_pred_hard))

print("\nConfusion Matrix - Soft Voting")
print(confusion_matrix(Y_test, y_pred_soft))

#---------------------------------------------------------------------------------------
# Step 7 : Visualization (model = hard, soft)
#---------------------------------------------------------------------------------------

model = ['hard voting', 'soft voting']
acciracies = [
    accuracy_score(Y_test, y_pred_hard),
    accuracy_score(Y_test, y_pred_soft)
]

plt.bar(model, acciracies, color=['blue', 'orange'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()
