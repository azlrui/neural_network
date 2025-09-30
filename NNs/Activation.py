import numpy as np
from .ActivationLayer import ActivationLayer
from .Layer import Layer

class ReLU(ActivationLayer):
    """
    Rectified Linear Unit activation:
        f(x) = max(0, x)

    - Forward: pass positive values unchanged, clip negatives to 0.
    - Backward: gradient is 1 where input > 0, else 0.
    """

    def __init__(self):
        super().__init__(
            activation=lambda x: np.maximum(0.0, x),
            activation_prime=lambda x: (x > 0).astype(x.dtype),
        )


class SoftMax(Layer):
    """
    Softmax activation over a given axis (default: features axis=1).

    Forward:
        probs = exp(x - max(x)) / sum(exp(x - max(x)))

    Backward:
        Given gradient wrt outputs (dL/ds), compute gradient wrt inputs (dL/dx)
        efficiently without building the full Jacobian:
            dL/dx = s * (g - <g, s>)

    Attributes
    ----------
    axis : int
        Axis along which softmax is applied.
    probs : np.ndarray | None
        Stores probabilities from forward() for reuse in backward().
    """

    def __init__(self, axis=1):
        self.axis = axis
        self.probs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities.

        Parameters
        ----------
        inputs : (N, M) array
            Raw scores (logits).

        Returns
        -------
        (N, M) array
            Probabilities along the given axis, each row summing to 1.
        """
        # Sanity check for axis
        if self.axis < 0 or self.axis >= inputs.ndim:
            raise ValueError(f"Invalid axis {self.axis} for input with shape {inputs.shape}")

        shifted = inputs - np.max(inputs, axis=self.axis, keepdims=True)
        exp_vals = np.exp(shifted)
        self.probs = exp_vals / np.sum(exp_vals, axis=self.axis, keepdims=True)

        # Sanity for probabilities
        sums = np.sum(self.probs, axis=self.axis)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise RuntimeError("Softmax probabilities do not sum to 1 (something went wrong! axis?)")

        return self.probs

    def backward(self, gradient_output: np.ndarray, learning_rate=None) -> np.ndarray:
        """
        Backpropagate gradient through softmax.

        Parameters
        ----------
        gradient_output : (N, M) array
            Gradient of loss wrt softmax outputs.
        learning_rate : ignored
            (Softmax has no trainable parameters).

        Returns
        -------
        (N, M) array
            Gradient of loss wrt softmax inputs.
        """
        if self.probs is None:
            raise RuntimeError("SoftMax.backward called before forward.")

        # Compute gradient
        dot = np.sum(gradient_output * self.probs, axis=self.axis, keepdims=True)
        grad_input = self.probs * (gradient_output - dot)

        # Sanity check of dimensions
        if grad_input.shape != gradient_output.shape:
            raise RuntimeError(
                f"Shape mismatch in SoftMax backward: "
                f"grad_input {grad_input.shape} vs grad_output {gradient_output.shape}"
            )

        return grad_input
