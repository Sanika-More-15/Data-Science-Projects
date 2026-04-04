import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#---------------------------------------------------------------------------------------
# Step 1 : Load the Dataset
#---------------------------------------------------------------------------------------

df = pd.read_csv("student-mat.csv", sep=";")

print(df.head())
print(df.shape)
print(df.isnull().sum())

#---------------------------------------------------------------------------------------
# Step 3 : Select Important Features
#---------------------------------------------------------------------------------------

features = df[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#---------------------------------------------------------------------------------------
# Step 3 : Feature Scaling
#---------------------------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

#---------------------------------------------------------------------------------------
# Step 4 : Appy K-means clustering
#---------------------------------------------------------------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

#---------------------------------------------------------------------------------------
# Step 5 : Check cluster Distribution
#---------------------------------------------------------------------------------------

print(df['cluster'].value_counts())

#---------------------------------------------------------------------------------------
# Step 6 : Analyze Clusters
#---------------------------------------------------------------------------------------

print(df.groupby('cluster')[["G1","G2","G3","studytime","failures","absences"]].mean())
# Which cluster = Top / Average / Struggling

#---------------------------------------------------------------------------------------
# Step 7 : Lable Clusters Meaningfully
#---------------------------------------------------------------------------------------

def label_student(cluster):
    if cluster == 0:
        return "Top Performers"
    elif cluster == 1:
        return "Average Students"
    else:
        return "Struggling Students"

df['cluster_label'] = df['cluster'].apply(label_student)

#---------------------------------------------------------------------------------------
# Step 8 : Visualization
#---------------------------------------------------------------------------------------

plt.scatter(df['G3'], df['absences'], c=df['cluster'])
plt.xlabel("Final Grade (G3)")
plt.ylabel("Absences")
plt.title("Student Clusters")
plt.show()

