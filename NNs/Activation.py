import numpy as np

class Activation():
    def forward(self, inputs): NotImplemented
    def backward(self, inputs, gradient_output): NotImplemented

class ReLU(Activation):
    def forward(self, inputs):
        return np.maximum(0.0, inputs)
    def backward(self, inputs, gradient_output):
        return gradient_output * (inputs > 0)

class SoftMax(Activation):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values)
        return probs
    def backward(self, inputs, gradient_output):
        return NotImplemented