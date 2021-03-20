import numpy as np
import nnfs
from dense_layer_class  import  Layer_Dense, Activation_ReLU, Activation_Softhmax, Loss, Loss_CategoricalCrossentropy
from nnfs.datasets import spiral_data

nnfs.init()

# create data set
x, y = spiral_data(samples=100, classes= 3)

#Create dense layer with 2 input features and 3 output values 
dense1 = Layer_Dense(2,3)

# Create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# Create second dense layer with 3 input features
# (as we take output of previus layer here) and 3 outputs values
dense2 = Layer_Dense(3,3)

# Create a Softmax activation( to be used with Dense layer)
activation2 = Activation_Softhmax()

# Create  loss function
loss_function = Loss_CategoricalCrossentropy()

# Performe a forward  pass of our training data through this layer
dense1.forward(x)

# forward pass through activation func.
# Takes in output from previus layer
activation1.forward(dense1.output)

# make a forward pass through second Dense layer
# it takes the output of second dense layer here
dense2.forward(activation1.output)


# Make a forward pass through activation funtion 
# it takes the output of second dense layer here
activation2.forward(dense2.output)


print(activation2.output[:5])

# Perform a forward pass through activation function
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output,y)

# print losss value
print('Loss: ', loss)