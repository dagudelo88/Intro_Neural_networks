import numpy as np
import nnfs
from dense_layer_class import Layer_Dense
from nnfs.datasets import spiral_data

nnfs.init()

# create data set

x, y = spiral_data(samples=100, classes= 3)

#Create dense layer with 2 input features and 3 output values 

dense1 = Layer_Dense(2,3)

# Performe a forward  pass of our training data through this layer

dense1.forward(x)

print(dense1.output[:5])
