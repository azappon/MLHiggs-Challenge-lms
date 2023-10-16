from typing import Tuple
import warnings
import numpy as np


def compute_mse(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """Compute the mean squared error.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray): weights. Shape: (d, )

    Returns:
        mse (float): mean squared error
    """

    # difference between the prediction and the real value
    e = y - tx @ w
    # mse of the residuals
    return (e**2).mean() / 2


def compute_mse_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the mean squared errror.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray): weights. Shape: (d, )

    Returns:
        gradient (np.ndarray): mean squared error gradient. Shape: (d, )
    """
    return -tx.T @ (y - tx @ w) / len(y)


def build_least_squares_matrices(
    y: np.ndarray, tx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute least squares matrices (for the system of normal equations).

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)

    Returns:
        A (np.ndarray): LHS matrix of normal equations. Shape: (d, d)
        b (np.ndarray): RHS matrix of normal equations. Shape: (d, )
    """

    A = tx.T @ tx
    b = tx.T @ y

    return A, b


def solve_least_squares_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute solution of system of normal equations.

    Args:
        A (np.ndarray): LHS matrix of normal equations. Shape: (d, d)
        b (np.ndarray): RHS matrix of normal equations. Shape: (d, )

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
    """

    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        warnings.warn(
            "The linear matrix equation is not well-determined to use np.solve(). "
            "Using np.lstsq() instead."
        )
        w = np.linalg.lstsq(A, b, rcond=None)[0]

    return w


def build_ridge_regression_matrices(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute regularized least squares matrices (for the system of ridge normal equations).

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        lambda_ (float): float

    Returns:
        A_ridge (np.ndarray): LHS matrix of ridge normal equations. Shape: (d, d)
        b (np.ndarray): RHS matrix of ridge normal equations. Shape: (d, )
    """

    A, b = build_least_squares_matrices(y, tx)
    A_ridge = A + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])

    return A_ridge, b


def get_prediction(tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the prediction of the model for binary classification in {0,1}.

    Args:
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray): weights. Shape: (d, )

    Returns:
        y_pred (np.ndarray): prediction of the model. Shape: (N, )
    """
    y_pred = (tx @ w) > 0.5
    return y_pred
