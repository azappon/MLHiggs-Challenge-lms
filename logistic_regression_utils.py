import numpy as np

from typing import Union


def sigmoid(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute sigmoid function at the given input.

    Args:
        t (float): input for the sigmoid function

    Returns:
        sigmoid (float): sigmoid output when input is t
    """

    return 1 / (1 + np.exp(-t))


def compute_nll(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.float_:
    """Compute negative log-likelihood.

    y is assumed to be in the interval {0, 1}.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray): weights. Shape: (d, )

    Returns:
        loss (np.float_): negative log-likelihood
    """
    N = y.shape[0]
    pred = tx @ w
    loss = -y.T.dot(pred) / N + np.mean(np.log(1 + np.exp(pred)))

    # Loss is a matrix (1, 1). We have to return only the scalar
    return np.float_(loss)


def compute_nll_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute negative log-likelihood gradient.

    y is assumed to be in the interval {0, 1}.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray): weights. Shape: (d, )

    Returns:
        gradient (np.ndarray): negative log-likelihood gradient. Shape: (d, )
    """

    N = y.shape[0]

    return 1 / N * tx.T @ (sigmoid(tx @ w) - y)


def compute_reg_nll_gradient(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float
) -> np.ndarray:
    """Compute regularized negative log-likelihood gradient.

    y is assumed to be in the interval {0, 1}.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray): weights. Shape: (d, )
        lambda_ (float): regularization parameter

    Returns:
        gradient (np.ndarray): regularized negative log-likelihood gradient. Shape: (d, )
    """

    return compute_nll_gradient(y=y, tx=tx, w=w) + 2 * lambda_ * w


def get_prediction(tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the prediction of the model for binary classification in {0,1}.

    Args:
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray): weights. Shape: (d, )

    Returns:
        y_pred (np.ndarray): prediction of the model. Shape: (N, )
    """
    y_pred = sigmoid(tx @ w) > 0.5
    return y_pred
