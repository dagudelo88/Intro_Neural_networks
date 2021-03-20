import numpy as np

softmax_output = np.array([[0.7,0.1,0.2],
                          [0.1,0.5,0.4],
                          [0.02,0.9,0.08]]) # estos serian los resultados de la funcion de activacion de la ultima capa usando el softmax

class_targets = np.array([[1,0,0],
                          [0,1,0],
                          [0,1,0]])  # estos son las classes de clasifcacion

# Probabilities for target values - only if categorical labels

if len(class_targets.shape) == 1:
    correct_confidences = softmax_output[range(len(softmax_output)),class_targets]


# Mask values - only for one-hot encoded labels

elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_output*class_targets,axis=1)

# losses

neg_log = -np.log(correct_confidences)

average_loss = np.mean(neg_log)

print(average_loss)


