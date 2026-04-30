import math
import numpy as np

# Mean Squared Error (MSE) Loss
def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Binary Cross Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

# Display the calculated loss
y_true_regression = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_regression = np.array([2.5, 0.0, 2.0, 8.0])

y_true_classification = np.array([1, 0, 1, 0])
y_pred_classification = np.array([0.9, 0.1, 0.8, 0.2])

mse_loss = mean_squared_error(y_true_regression, y_pred_regression)
bce_loss = binary_cross_entropy(y_true_classification, y_pred_classification)
print("Mean Squared Error Loss:", mse_loss)
print("Binary Cross Entropy Loss:", bce_loss)
# Explanation of loss functions
print("\nThe Mean Squared Error (MSE) loss is used for regression problems, where the goal is to predict continuous values. It measures the average squared difference between the actual and predicted values." \
      "\nThe Binary Cross Entropy Loss is used for binary classification problems, where the goal is to predict probabilities of class membership. It measures the difference between the actual binary labels and the predicted probabilities.")