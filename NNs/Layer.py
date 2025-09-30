class Layer:
    """
    Abstract base class for all network layers.

    Any concrete layer (Dense, ReLU, SoftMax, etc.) must implement:
      - forward(): how the layer transforms inputs into outputs
      - backward(): how the layer propagates gradients backward,
                    and optionally updates its own parameters
    """

    def forward(self, inputs):
        """
        Compute the forward pass through this layer.

        Parameters
        ----------
        inputs : np.ndarray
            Input data or activations from the previous layer,
            shape depends on the layer type.

        Returns
        -------
        np.ndarray
            Output of this layer, to be passed to the next layer.
        """
        raise NotImplementedError("forward() must be implemented by subclasses")

    def backward(self, grad_output, learning_rate=None):
        """
        Compute the backward pass (backpropagation).

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of the loss w.r.t. the layer's output,
            received from the next layer in the network.
        learning_rate : float, optional
            Learning rate for parameter update if this layer
            has trainable weights (e.g., DenseLayer). Ignored
            for layers without parameters.

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. this layer's inputs,
            to be passed to the previous layer.
        """
        raise NotImplementedError("backward() must be implemented by subclasses")
