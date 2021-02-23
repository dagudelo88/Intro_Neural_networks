import numpy as np

# values from earlier previus when we described
# what a neural network is
layer_outputs =[4.8,1.21,2.385]

# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('exponential values: ')
print(exp_values)


# Now normalice values
norm_values = exp_values / np.sum(exp_values)

print('Normaliced exponentiated values: ')
print(norm_values)
print(' Sum of normalized values: ', np.sum(norm_values))

