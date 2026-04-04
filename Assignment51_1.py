import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0 # fake news
true_df["label"] = 1 # true news

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True) # shuffle the dataset

print(df.head())
print(df.shape)
print(df.isnull().sum())

df["content"] = df["title"] + " " + df["text"]
df = df.dropna()



# Use TF-IDF to convert text data into numerical features
# Now concatenate then into one dataframe.
# You may use either title, text, or both combined 
#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------