import numpy as np 

# dense layer

class Layer_Dense:
    # Layer initialization
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
# ReLU activation
class Activation_ReLU:
    
    #forward pass
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

        