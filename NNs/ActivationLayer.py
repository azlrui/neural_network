import numpy as np
from .Layer import Layer
from typing import Callable, Optional

class ActivationLayer(Layer):
    """
    Generic element-wise activation wrapper.

    Assumptions
    ----------
    - `activation` and `activation_prime` are vectorized functions applied
      element-wise and return arrays of the *same shape* as the input.
    - This layer has no trainable parameters; `learning_rate` is ignored.

    Notes
    -----
    Use a *dedicated* layer (not this one) for non element-wise activations
    with coupled dimensions (e.g., Softmax). Those need custom backward logic.
    """

    def __init__(self, activation: Callable[[np.ndarray], np.ndarray],
                 activation_prime: Callable[[np.ndarray], np.ndarray]):
        if not callable(activation) or not callable(activation_prime):
            raise TypeError("activation and activation_prime must be callables.")
        
        self.activation_function = activation
        self.activation_prime = activation_prime
        self.inputs: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply activation element-wise and cache inputs for backward.

        Parameters
        ----------
        inputs : np.ndarray
            Activations from the previous layer. Any shape is allowed.

        Returns
        -------
        np.ndarray
            Activated outputs with the same shape as `inputs`.
        """
        self.inputs = inputs

        out = self.activation_function(inputs)

        # Sanity checks for type and shapes
        if not isinstance(out, np.ndarray):
            raise TypeError("activation_function must return a numpy array.")
        if out.shape != inputs.shape:
            raise ValueError(
                f"Activation output shape {out.shape} does not match input shape {inputs.shape}."
            )
        return out

    def backward(self, output_gradient: np.ndarray, learning_rate: float | None = None) -> np.ndarray:
        """
        Chain rule for element-wise activations:
            dL/dX = dL/dy ⊙ f'(x)

        Parameters
        ----------
        output_gradient : np.ndarray
            Gradient of loss w.r.t. the activation outputs (same shape as inputs).
        learning_rate : float | None
            Ignored (no parameters in this layer).

        Returns
        -------
        np.ndarray
            Gradient of loss w.r.t. the activation inputs (same shape as inputs).
        """
        
        # Sanity checks for sequence of operations and shape
        if self.inputs is None:
            raise RuntimeError("ActivationLayer.backward called before forward.")
        
        if output_gradient.shape != self.inputs.shape:
            raise ValueError(
                f"output_gradient shape {output_gradient.shape} "
                f"does not match cached input shape {self.inputs.shape}."
            )

        deriv = self.activation_prime(self.inputs)

        # Sanity checks for type and shapes
        if not isinstance(deriv, np.ndarray):
            raise TypeError("activation_prime must return a numpy array.")
        if deriv.shape != self.inputs.shape:
            raise ValueError(
                f"activation_prime returned shape {deriv.shape}, "
                f"expected {self.inputs.shape}."
            )

        # Element-wise product implements the chain rule
        grad_input = output_gradient * deriv

        return grad_input
