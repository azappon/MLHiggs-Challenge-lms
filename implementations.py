"""This file contains implementations of the baseline optimization algorithms"""
from typing import Tuple
import numpy as np

import dataset_utils
import linear_regression_utils
import logistic_regression_utils

RANDOM_SEED = 1


def mean_squared_error_gd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, np.float_]:
    """Linear regression using gradient descent with the mean squared error cost function.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        initial_w (np.ndarray): initial weights. Shape: (d, )
        max_iters (int): maximum number of iterations.
        gamma (float): learning rate.

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
        final_loss (np.float_): MSE loss computed for optimized weights.
    """

    # initialize weight vector
    w = initial_w

    # repeat the following steps until we reach the maximum number of iterations
    for _ in range(max_iters):
        # gradient of the loss function for the current weight vector w
        grad = linear_regression_utils.compute_mse_gradient(y=y, tx=tx, w=w)
        # update the weight vector w by subtracting gamma times the gradient
        w = w - gamma * grad

    # return the final weight vector w, and the corresponding loss function value
    final_loss = linear_regression_utils.compute_mse(y, tx, w)

    return w, final_loss


def mean_squared_error_sgd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, np.float_]:
    """Linear regression using stochastic gradient descent with the mean squared error cost function.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        initial_w (np.ndarray): initial weights. Shape: (d, )
        max_iters (int): maximum number of iterations.
        gamma (float): learning rate.

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
        final_loss (np.float_): MSE loss computed for optimized weights.
    """

    # initialize weight vector
    w = initial_w

    # Create sample generator
    sample_generator = dataset_utils.batch_generator(
        y=y,
        tx=tx,
        batch_size=1,
        num_batches=max_iters,
        seed=RANDOM_SEED,
    )

    for _ in range(max_iters):
        sample_y, sample_tx = next(sample_generator)
        gradient = linear_regression_utils.compute_mse_gradient(
            y=sample_y, tx=sample_tx, w=w
        )
        w -= gamma * gradient

    final_loss = linear_regression_utils.compute_mse(y=y, tx=tx, w=w)

    return w, final_loss


def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[np.ndarray, np.float_]:
    """Least squares regression using normal equations.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
        final_loss (np.float_): MSE loss computed for optimized weights.
    """

    A, b = linear_regression_utils.build_least_squares_matrices(y, tx)
    w = linear_regression_utils.solve_least_squares_system(A, b)

    final_loss = linear_regression_utils.compute_mse(y=y, tx=tx, w=w)

    return w, final_loss


def ridge_regression(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> Tuple[np.ndarray, np.float_]:
    """Ridge regression using normal equations.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        lambda_ (float): regularization parameter.

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
        final_loss (np.float_): MSE loss computed for optimized weights.
    """

    A_ridge, b = linear_regression_utils.build_ridge_regression_matrices(y, tx, lambda_)
    w = linear_regression_utils.solve_least_squares_system(A_ridge, b)

    final_loss = linear_regression_utils.compute_mse(y, tx, w)

    return w, final_loss


def logistic_regression(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, np.float_]:
    """Logistic regression using gradient descent.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        initial_w (np.ndarray): initial weights. Shape: (d, )
        max_iters (int): maximum number of iterations.
        gamma (float): learning rate.

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
        final_loss (np.float_): negative log-likelihood loss computed for optimized weights.
    """

    w = initial_w

    # Run full gradient descent
    for _ in range(max_iters):
        grad = logistic_regression_utils.compute_nll_gradient(y=y, tx=tx, w=w)
        w -= gamma * grad

    # Compute final loss
    final_loss = logistic_regression_utils.compute_nll(y=y, tx=tx, w=w)

    return w, final_loss


def logistic_regression_sgd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, np.float_]:
    """Logistic regression using stochastic gradient descent.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        initial_w (np.ndarray): initial weights. Shape: (d, )
        max_iters (int): maximum number of iterations.
        gamma (float): learning rate.

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
        final_loss (np.float_): negative log-likelihood loss computed for optimized weights.
    """

    w = initial_w

    # Create sample generator
    sample_generator = dataset_utils.batch_generator(
        y=y,
        tx=tx,
        batch_size=1,
        num_batches=max_iters,
        seed=RANDOM_SEED,
    )

    # Run stochastic gradient descent
    for _ in range(max_iters):
        sample_y, sample_tx = next(sample_generator)
        grad = logistic_regression_utils.compute_nll_gradient(
            y=sample_y, tx=sample_tx, w=w
        )
        w -= gamma * grad

    # Compute final loss
    final_loss = logistic_regression_utils.compute_nll(y=y, tx=tx, w=w)

    return w, final_loss


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, np.float_]:
    """Regularized logistic regression using gradient descent.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        lambda_ (float): regularization parameter.
        initial_w (np.ndarray): initial weights. Shape: (d, )
        max_iters (int): maximum number of iterations.
        gamma (float): learning rate.

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
        final_loss (np.float_): negative log-likelihood loss computed for optimized weights.
    """

    w = initial_w

    # Run full gradient descent
    for _ in range(max_iters):
        grad = logistic_regression_utils.compute_reg_nll_gradient(
            y=y,
            tx=tx,
            w=w,
            lambda_=lambda_,
        )
        w -= gamma * grad

    # Compute final loss
    final_loss = logistic_regression_utils.compute_nll(y=y, tx=tx, w=w)

    return w, final_loss


def reg_logistic_regression_sgd(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, np.float_]:
    """Regularized logistic regression using stochastic gradient descent.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        lambda_ (float): regularization parameter.
        initial_w (np.ndarray): initial weights. Shape: (d, )
        max_iters (int): maximum number of iterations.
        gamma (float): learning rate.

    Returns:
        w (np.ndarray): optimized weights. Shape: (d, )
        final_loss (np.float_): negative log-likelihood loss computed for optimized weights.
    """
    w = initial_w

    # Create sample generator
    sample_generator = dataset_utils.batch_generator(
        y=y,
        tx=tx,
        batch_size=1,
        num_batches=max_iters,
        seed=RANDOM_SEED,
    )

    # Run stochastic gradient descent
    for _ in range(max_iters):
        # Get batch (with only one element)
        sample_y, sample_tx = next(sample_generator)
        # Gradient step
        grad = logistic_regression_utils.compute_reg_nll_gradient(
            y=sample_y,
            tx=sample_tx,
            w=w,
            lambda_=lambda_,
        )
        w -= gamma * grad

    # Compute final loss
    final_loss = logistic_regression_utils.compute_nll(y=y, tx=tx, w=w)

    return w, final_loss
