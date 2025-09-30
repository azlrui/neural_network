import numpy as np
from .Layer import Layer

class DenseLayer(Layer):
    """
    Fully connected (dense) layer:
        Y = XW + b

    Parameters
    ----------
    input_size : int
        Number of features in the input (D).
    output_size : int
        Number of units / features in the output (M).
    seed : int, optional
        Random seed for reproducible initialization.
    """

    def __init__(self, input_size, output_size, seed=42):
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        # TODO: implement more sophisticated initialization parameters
        self.weights = rng.normal(size=(input_size, output_size))
        self.biases = np.zeros((1, output_size))
        self.inputs = None  # cache inputs for backward pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute outputs.

        Parameters
        ----------
        inputs : (N, D) array
            Batch of inputs (N samples, D features).

        Returns
        -------
        (N, M) array
            Outputs (N samples, M features).
        """
        # Weights sanity check
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Input has {inputs.shape[1]} features, "
                f"but weights expect {self.weights.shape[0]}"
            )

        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.biases

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass: propagate gradient and update parameters.

        Parameters
        ----------
        output_gradient : (N, M) array
            Gradient of loss w.r.t. this layer's outputs. (dL / dY)
        learning_rate : float
            Step size for gradient descent.

        Returns
        -------
        (N, D) array
            Gradient of loss w.r.t. this layer's inputs.
        """
        N, D = self.inputs.shape
        _, M = output_gradient.shape

        # Gradient Sanity check
        if M != self.weights.shape[1]:
            raise ValueError(
                f"output_gradient has {M} features, "
                f"but weights produce {self.weights.shape[1]}"
            )

        # Compute gradients
        dL_dW = self.inputs.T @ output_gradient       # (D, N) @ (N, M) = (D, M)
        dL_db = output_gradient.sum(axis=0, keepdims=True)  # (1, M)
        dL_dX = output_gradient @ self.weights.T      # (N, M) @ (M, D) = (N, D)

        # Parameter update
        self.weights -= learning_rate * dL_dW
        self.biases  -= learning_rate * dL_db

        return dL_dX

# import numpy as np 
# from .Layer import Layer 

# class DenseLayer(Layer): 
#     def __init__(self, input_size, output_size, seed=42): 
#         rng = np.random.default_rng(seed) 
#         self.weights = np.random.rand(input_size, output_size) # TODO: implement method to activate 
#         self.biases = np.zeros((1,output_size)) 
#         self.inputs = None 
        
#     def forward(self, inputs): 
#         self.inputs = inputs 
#         return np.dot(self.inputs, self.weights) + self.biases # returns Y 
    
#     def backward(self, output_gradient, learning_rate): # compute the gradients dL/dW , dL/db , dL/dX 
#         dL_dW = self.inputs.T @ output_gradient # computes the derivative of the loss wrt the weights (i.e gradient) 
#         dL_db = output_gradient.sum(axis=0, keepdims=True) # compute the derivative of the loss wrt to the bias 
#         dL_dX = output_gradient @ self.weights.T # compute the derivative of the loss wrt to the input 
        
#         # compute the new weights and biases 
#         self.weights = self.weights-learning_rate*dL_dW # TODO: extend gradient descent (ADAM?) 
#         self.biases = self.biases-learning_rate*dL_db # TODO: extend gradient descent 
#         return dL_dX