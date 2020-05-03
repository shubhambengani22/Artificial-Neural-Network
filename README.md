# Artificial-Neural-Network
A simple Neural Network for multiple hidden layers

The inspiration and parts of code are adapted from deeplearning.ai course "Neural Network and Deep Learning", but it has been generalized for many hidden layers.

The implementation done is for the Kaggle competetion "Dogs vs Cats" and trained for images.

Input X(Set of features for supervised learning) - numpy array
Y - numpy array of shape : (1, num_examples)

Hyperparameters to play around :

-> Layer_dims : A list of the number of units in each layer when the length of the list signifies the number of layers and the input features. Also, each element in the list signifies the number of units in the corresponding layer.

-> Learning rate : Changed according to the dataset, lowered or increased according the change of the gradients.

-> Number of iterations : This is an integer specifiying the number of iterations to train the network on the dataset.
