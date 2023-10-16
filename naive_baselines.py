from typing import Tuple

import numpy as np


def random_guess(y: np.ndarray, tx: np.ndarray, seed=10) -> Tuple[int, np.float_]:
    """Random guess algorithm.

    Args:
        y (np.ndarray): dataset labels. Shape: (d, )
        tx (np.ndarray): dataset samples. Shape: (N, d)
        seed (int, optional): random seed. Defaults to 10.

    Returns:
        Tuple[int, np.float_]: final weights (dummy) and corresponding loss.
    """

    w = seed
    rng = np.random.default_rng(seed)
    y_pred = rng.choice([0, 1], size=y.shape)
    final_loss = np.mean(y == y_pred).squeeze()

    return w, final_loss


def majority_guess(y: np.ndarray, tx: np.ndarray) -> Tuple[int, np.float_]:
    """Majority guess algorithm.

    Args:
        y (np.ndarray): dataset labels. Shape: (d, )
        tx (np.ndarray): dataset features. Shape: (N, d)

    Returns:
        Tuple[int, np.float_]: final weights (dummy) and corresponding loss.
    """

    w = (np.mean(y) > 0.5).squeeze()
    final_loss = np.mean(y == w).squeeze()

    return w, final_loss


def get_random_prediction(tx: np.ndarray, w: int) -> np.ndarray:
    """Compute a random prediction for binary classification in {0,1}.

    Args:
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray: weights (dummy). Shape: (d, )

    Returns:
        y_pred (np.ndarray): prediction of the model. Shape: (N, )
    """
    rng = np.random.default_rng(w)
    y_pred = rng.choice([0, 1], size=tx.shape[0])
    return y_pred


def get_majority_prediction(tx: np.ndarray, w: int) -> np.ndarray:
    """Compute a majority prediction for binary classification in {0,1}.

    Args:
        tx (np.ndarray): features. Shape: (N, d)
        w (np.ndarray: weights (dummy). Shape: (d, )

    Returns:
        y_pred (np.ndarray): prediction of the model. Shape: (N, )
    """
    y_pred = np.ones(tx.shape[0]) * w
    return y_pred
