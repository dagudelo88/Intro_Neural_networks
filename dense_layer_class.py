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

# Common loss class
class Loss:
    # caluclates the data and regularization losses
    # given modeloutput and ground truth values
    def calculate(self, output,y):
        # calculate sample losses
        sample_losses = self.forward(output,y)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        # return loss
        return data_loss

# cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, Y_pred, y_true):

        # Number of samples in a batch
        samples = len(Y_pred)

        # clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        Y_pred_clipped = np.clip(Y_pred,1e-7, 1-1e-7)

        # probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = Y_pred_clipped[range(samples),y_true]

        # mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(Y_pred_clipped*y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    
