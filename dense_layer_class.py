# dense layer

class Layer_Dense:
    # Layer initialization
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    # Forward pass
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights)+self.biases
        pass
