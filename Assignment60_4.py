import math
import numpy as np

# Initialize parameters
input_value = 0.5
w1 = 0.4
w2 = 0.6
bias = 0.5
target_output = 1
learning_rate = 0.01

# Calculate weighted sum and prediction
weighted_sum = w1 * input_value + w2 * input_value + bias
predicted_output = 1 / (1 + np.exp(-weighted_sum))  # Sigmoid activation function
print("Old weights: w1 =", w1, ", w2 =", w2)
print("Predicted output (sigmoid func):", predicted_output)

# Calculate error
error = target_output - predicted_output
print("Error:", error)

# Update weights using gradient descent logic
w1 = w1 + learning_rate * error * input_value
w2 = w2 + learning_rate * error * input_value
bias = bias + learning_rate * error
print("Updated weights: w1 =", w1, ", w2 =", w2, ", bias =", bias)

