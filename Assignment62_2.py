import math
import numpy as np

feature_map = np.array([[1, -2, 3],[-4, 5, -7],[6, -7, 8]])
print("\nFeature Map with Positive and Negative Values:")
print(feature_map)

# Apply ReLU
relu_output = np.maximum(0, feature_map)
print("\nFeature Map after ReLU Activation:")
print(relu_output)

# Apply 2x2 Max Pooling
pooled_output = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        pooled_output[i][j] = np.max(relu_output[i:i+2, j:j+2])
print("\nFeature Map after 2x2 Max Pooling:")
print(pooled_output)

# Explanation of pooling reducing size
print("\nPooling reduces the size of the feature map by summarizing the information in a local region (e.g., 2x2) into a single value (the maximum in this case). " \
"\nThis helps to reduce the computational load and also makes the model more robust to small translations in the input image, as it captures the most important features while discarding less relevant details.")
