import numpy as np
import nnfs
from dense_layer_class import Layer_Dense, Activation_ReLU
from nnfs.datasets import spiral_data

nnfs.init()

# create data set

x, y = spiral_data(samples=100, classes= 3)

#Create dense layer with 2 input features and 3 output values 

dense1 = Layer_Dense(2,3)

# Create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# Performe a forward  pass of our training data through this layer
dense1.forward(x)

# forward pass through activation func.
# Takes in output from previus layer
activation1.forward(dense1.output)

print(activation1.output[:5])
