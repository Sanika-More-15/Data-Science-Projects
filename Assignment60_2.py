import numpy
import matplotlib.pyplot as plt

# Accept input values from -10 to 10 and calculate the outputs for each activation function
input_value = float(input("Enter an input value between -10 and 10: "))

w1 = 0.4
w2 = 0.6
bias = 0.5

# Calculate weightes sum ( x1* w1 + x2*w2 ) + bias
weighted_sum = (input_value * w1) + (input_value * w2) + bias

print( " Weighted sum :", weighted_sum)

# Calculate output using sigmoid activation function
sigmoid_output = 1 / (1 + numpy.exp(-weighted_sum))
print(" Sigmoid output :", sigmoid_output)

ReLU_output = max(0, weighted_sum)
print(" ReLU output :", ReLU_output)

tanh_output = (numpy.exp(weighted_sum) - numpy.exp(-weighted_sum)) / (numpy.exp(weighted_sum) + numpy.exp(-weighted_sum))
print(" Tanh output :", tanh_output)

# Plot the activation functions
x = numpy.linspace(-10, 10, 100)
sigmoid = 1 / (1 + numpy.exp(-x))
ReLU = numpy.maximum(0, x)
tanh = (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(x, sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.subplot(1, 3, 2)
plt.plot(x, ReLU, label='ReLU')
plt.title('ReLU Activation Function')
plt.subplot(1, 3, 3)
plt.plot(x, tanh, label='Tanh')
plt.title('Tanh Activation Function')
plt.tight_layout()
plt.show()

# Explain the use of each activation function
print(" The sigmoid activation function is used to map the input values to a range between 0 and 1, which is useful for binary classification problems. " \
"The ReLU activation function is used to introduce non-linearity into the model and is commonly used in hidden layers of neural networks." \
" The tanh activation function is similar to sigmoid but maps the input values to a range between -1 and 1, which can be useful for certain types of data and can help with convergence during training.")