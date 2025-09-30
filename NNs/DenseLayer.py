import numpy as np
from Layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) # TODO: implement method to activate
        self.biases = np.random.rand((1,output_size))
    
    def forward(self, inputs):
        self.input = inputs
        return np.dot(self.input, self.weights) + self.biases # returns Y
    
    def backward(self, output_gradient, learning_rate):
        self.gradient_weights = np.dot(output_gradient, self.input.T) # computes the derivative of the loss wrt the weights (i.e gradient)
        self.weights = self.weights - learning_rate * self.gradient_weights # TODO: extend gradient descent (ADAM?)
        self.bias = self.bias - learning_rate * output_gradient # TODO: extend gradient descent
        return np.dot(self.weights.T, output_gradient) # compute the derivative of the loss wrt to the input