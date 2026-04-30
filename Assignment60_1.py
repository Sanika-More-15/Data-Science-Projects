import numpy

x1 = 2
x2 = 3
w1 = 0.4
w2 = 0.6
bias = 0.5

# Calculate weightes sum ( x1* w1 + x2*w2 ) + bias
weighted_sum = (x1 * w1) + (x2 * w2) + bias
print( " Weighted sum :", weighted_sum)

# Calculate output using sigmoid activation function
sigmoid_output = 1 / (1 + numpy.exp(-weighted_sum))
print(" Sigmoid output :", sigmoid_output)

# Expalin whether the output is closer to 0 or 1
print(" The output is closer to 1 than 0, which indicates that the neuron is more likely to be activated based on the given inputs and weights.")