import numpy as np
from .Activation import ReLU, SoftMax
from .DenseLayer import DenseLayer
from .ActivationLayer import ActivationLayer  # only for typing / clarity
from typing import List, Optional, Sequence, Tuple

# Contains all the implemented activation functions
_ACT = {
    None:      lambda: None,
    "relu":    lambda: ReLU(),
    "softmax": lambda: SoftMax(axis=1),
}

def _check_lists_same_length(dense_layers_sizes: Sequence[Tuple[int, int]],
                             activation_layers: Sequence[Optional[str]]) -> None:
    """Ensure there is exactly one activation spec per Dense layer."""
    if len(dense_layers_sizes) != len(activation_layers):
        raise ValueError("dense_layers_sizes and activations must have the same length.")

def _check_adjacent_dims(layer_specs: Sequence[Tuple[int, int]]) -> None:
    """Ensure out_dim of layer i matches in_dim of layer i+1."""
    for i in range(len(layer_specs) - 1):  # last has no successor to check
        out_dim = layer_specs[i][1]
        next_in = layer_specs[i + 1][0]
        if out_dim != next_in:
            raise ValueError(
                f"Size mismatch: layer {i} out_dim={out_dim} != "
                f"layer {i+1} in_dim={next_in}"
            )

def _normalize_and_validate_activations(acts: Sequence[Optional[str]]) -> List[Optional[str]]:
    """Lowercase, allow None, and validate names against the factory."""
    normed: List[Optional[str]] = []
    for i, a in enumerate(acts):
        name = None if a is None else str(a).lower()
        if name not in _ACT:
            valid = ", ".join([str(k) for k in _ACT.keys()])
            raise ValueError(f"Unknown activation '{a}' at index {i}. Valid: {valid}")
        normed.append(name)
    return normed
        

class NeuralNetwork:
    """
    Minimal sequential NN with Dense + Activation pairs.

    Parameters
    ----------
    dense_layers_sizes : Sequence[Tuple[int, int]]
        List of (in_dim, out_dim) for each Dense layer, e.g. [(4, 8), (8, 3)].
    activation_layers : Sequence[Optional[str]]
        List of activation names after each Dense layer. Use one of:
        {None, 'relu', 'softmax'}. Must be same length as dense_layers_sizes.
    learning_rate : float
        SGD learning rate used by Dense layers during backward().
    seed : int | None
        Optional seed; we offset by layer index to avoid identical draws.

    Notes
    -----
    - This class updates parameters *inside* backward(). If you later add
      optimizers (Adam, momentum), consider splitting "compute grads" from
      "apply step".
    """
    def __init__(
        self,
        dense_layers_sizes: Sequence[Tuple[int, int]],
        activation_layers: Sequence[Optional[str]],
        learning_rate: float = 1e-2,
        seed: Optional[int] = 42,
    ):
        # Sanity check
        _check_lists_same_length(dense_layers_sizes, activation_layers)
        _check_adjacent_dims(dense_layers_sizes)
        acts = _normalize_and_validate_activations(activation_layers)

        # Initialization
        self.learning_rate = float(learning_rate)
        self.layer_specs: List[Tuple[int, int]] = list(dense_layers_sizes)
        self.layers: List[object] = []

        # Build [Dense -> (Activation?)] network sequence
        for i, (in_dim, out_dim) in enumerate(self.layer_specs):
            layer_seed = None if seed is None else seed + i
            self.layers.append(DenseLayer(in_dim, out_dim, seed=layer_seed))
            act = _ACT[acts[i]]()
            if act is not None:
                self.layers.append(act)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run a forward pass through all layers.

        Parameters
        ----------
        inputs : (N, D) array

        Returns
        -------
        (N, M_L) array
            Output of the final layer.
        """
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass through the network.

        Parameters
        ----------
        output_grad : (N, M_L) array
            Gradient of the loss w.r.t. the final output of the network.

        Returns
        -------
        (N, D) array
            Gradient of the loss w.r.t. the original inputs.
        """
        grad = output_grad
        for layer in reversed(self.layers):
            # Dense layers require lr to update parameters; activations do not.
            if isinstance(layer, DenseLayer):
                grad = layer.backward(grad, self.learning_rate)
            else:
                grad = layer.backward(grad, None)
        return grad
