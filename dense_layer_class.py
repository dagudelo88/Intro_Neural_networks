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


# Softmax activation
class Activation_Softhmax:
    # Forward pass
    def forward(self,inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 

        # Normalized them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities    