import numpy as np
from sklearn.preprocessing import StandardScaler

X_train=[[25, 20000],
      [30, 40000],
      [35, 80000]]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

print("X_train data is :")
print(X_train)
print("X_train_scaled is :")
print(X_train_scaled)