import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

Point1=[[10, 2, 9]]
Point2=[[12, 1, 6]]

data=np.vstack([Point1, Point2])

distance_matrix = euclidean_distances(Point1, Point2)
print("Euclidean distance before feature scaling :",distance_matrix[0][0])

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

scaled_point1 = scaled_data[0]
scaled_point2 = scaled_data[1]

print("\nOriginal Point 1:", Point1)
print("Scaled Point 1:", scaled_point1)
print("Original Point 2:", Point2)
print("Scaled Point 2:", scaled_point2)

dis_matrix = euclidean_distances(scaled_point1.reshape(1,-1), scaled_point2.reshape(1,-1))
print("\nEuclidean distance after feature scaling :",dis_matrix[0][0])