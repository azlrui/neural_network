import numpy as np
from NNs.NN import NeuralNetwork  # assumes your package layout is correct

EPS = 1e-12  # numerical stability

# Losses & Gradients functions
def cross_entropy_loss(y_true, probs):
    """
    Mean cross-entropy for one-hot labels and probabilities.
    y_true: (N, C) one-hot
    probs : (N, C) softmax probabilities
    returns: scalar
    """
    return -np.mean(np.sum(y_true * np.log(probs + EPS), axis=1))

def cross_entropy_grad_wrt_softmax_output(y_true, probs):
    """
    Gradient of CE wrt softmax OUTPUTS (dL/ds), averaged over batch.
    Use this ONLY if network's last layer is SoftMax.
    """
    N = probs.shape[0]
    return -(y_true / (probs + EPS)) / N

def mse_loss(y_true, y_pred):
    """Standard MSE (mean over batch)."""
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_true, y_pred):
    """Gradient wrt predictions for MSE, averaged over batch."""
    N = y_pred.shape[0]
    return - (2.0 / N) * (y_true - y_pred)

# Softmax function
def softmax(logits, axis=1):
    """
    Numerically stable softmax.
    
    Parameters
    ----------
    logits : np.ndarray
        Raw scores (N, C).
    axis : int
        Axis to normalize (default: 1 for classes).
    
    Returns
    -------
    probs : np.ndarray
        Softmax probabilities (same shape as logits).
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    expv = np.exp(shifted)
    return expv / np.sum(expv, axis=axis, keepdims=True)

# Utils
def make_spiral_data(n_points=600, n_classes=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    N = n_points // n_classes
    X = np.zeros((n_points, 2))
    y = np.zeros(n_points, dtype=np.int32)

    for j in range(n_classes):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + rng.normal(0, noise, N)  # theta with noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # one-hot labels
    Y = np.eye(n_classes)[y]
    return X, y, Y

def main():
    rng = np.random.default_rng(0)

    # Toy example
    X, y_idx, Y = make_spiral_data(n_points=600, n_classes=3, noise=0.3, seed=42)


    # Keep SoftMax in the model; use CE gradient wrt softmax outputs
    activation_layers = ["relu", "softmax", "softmax"]
    
    net = NeuralNetwork(
        dense_layers_sizes=[(X.shape[1], 29), (29, 17), (17, Y.shape[1])],
        activation_layers=activation_layers,
        learning_rate=0.6,
        seed=123
    )

    for epoch in range(1, 501):
        out = net.forward(X)

        if activation_layers[-1].lower() == "softmax":
            # Network has already a last layer softmax
            probs = out
            loss = cross_entropy_loss(Y, probs)
            grad = cross_entropy_grad_wrt_softmax_output(Y, probs)  # dL/ds
        else:
            # No softmax in network -> `out` are logits
            logits = out
            probs = softmax(logits)  # just for loss/metrics reporting

            loss = cross_entropy_loss(Y, probs)
            grad = (probs - Y) / X.shape[0]  # CE+Softmax shortcut dL/dlogits

        net.backward(grad)

        if epoch % 50 == 0:
            preds = np.argmax(probs, axis=1)
            acc = (preds == y_idx).mean()
            print(f"epoch {epoch:3d}  loss={loss:.4f}  acc={acc:.3f}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())