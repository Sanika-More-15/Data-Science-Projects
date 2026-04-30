import math
import numpy as np

image = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]
image = np.array(image)
print("\nOriginal 5x5 Image")
print(image)

kernel = [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
]
kernel = np.array(kernel)
print("\n 3x3 Kernel")
print(kernel)

feature_map = np.zeros((3, 3))

# Convolution Operation
# Output Size = (5-3+1) x (5-3+1) = 3x3

feature_map = np.zeros((3,3))

for i in range(3):
    for j in range(3):

        # Extract 3x3 region
        region = image[i:i+3, j:j+3]

        # Multiply and Sum
        result = np.sum(region * kernel)

        # Store result
        feature_map[i][j] = result
        
        # Printing each step as requested
        print(f"\nRegion at ({i},{j}):\n{region}")
        print(f"Calculation Result: {result}")

# Print each region calculation
print("\nFeature Map (3x3) after Convolution:")
print(feature_map)
