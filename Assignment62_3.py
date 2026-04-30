import math
import numpy as np

# Create a 2D feature map
feature_map = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nOriginal 2D Feature Map:")
print(feature_map)

# Flatten the feature map
flattened = feature_map.flatten()
print("\nFlattened Feature Map:")
print(flattened)

# Define weights and bias for the fully connected layer
weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
bias = 1.0

# Calculate the output of the fully connected layer
output = np.dot(flattened, weights) + bias
print("\nOutput of the Fully Connected Layer:")
print(output)

# Explanation of the role of flatten layer in CNN
print("\nThe flatten layer in a Convolutional Neural Network (CNN) serves the purpose of converting the multi-dimensional feature maps produced by convolutional and pooling layers into a one-dimensional vector. " \
"\nThis is necessary because the fully connected layers that follow require a flat input to perform their computations. " \
"\nThe flatten layer essentially takes all the features extracted by the previous layers and prepares them for classification or regression tasks by creating a single long vector that can be fed into the fully connected layer.")