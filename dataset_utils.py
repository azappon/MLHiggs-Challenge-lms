from typing import List, Tuple
import numpy as np

D_DIM = 1


def batch_generator(
    y: np.ndarray, tx: np.ndarray, batch_size: int, num_batches: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a minibatch iterator for a dataset.

    Args:
        y (np.ndarray): labels. Shape: (N, )
        tx (np.ndarray): features. Shape: (N, d)
        batch_size (int): size of a batch
        num_batches (int): number of batches
        seed (int): random seed (fixed for reproducibility)

    Yields:
        sampled_y (np.ndarray): batch labels
        sample_tx (np.ndarray): batch features
    """

    # data_size is the total number of samples in the dataset.
    data_size = len(y)

    # Get random sampling of the dataset
    rng = np.random.default_rng(seed)

    # iterate through the wanted number of batches
    for _ in range(num_batches):
        sample_idxs = rng.integers(low=0, high=data_size, size=batch_size)
        # yield the current batch
        sampled_y = y[sample_idxs]
        sampled_tx = tx[sample_idxs]
        yield sampled_y, sampled_tx


def from_01_to_11(y: np.ndarray) -> np.ndarray:
    """Transform y from the interval {0, 1} to the interval {-1, 1}.

    Args:
        y (np.ndarray): labels. Shape: (N, )

    Returns:
        y (np.ndarray): labels. Shape: (N, )
    """

    return 2 * y - 1


def from_11_to_01(y: np.ndarray) -> np.ndarray:
    """Transform y from the interval {-1, 1} to the interval {0, 1}.

    Args:
        y (np.ndarray): labels. Shape: (N, )

    Returns:
        y (np.ndarray): labels. Shape: (N, )
    """

    return (y + 1) / 2


def get_categorical_idx(x_train: np.ndarray) -> int:
    """Gets the idx to the categorical variable. Hard coded for JET_NUM which
    has 4 values (it assumes x_train is freshly loaded from the csv file.).

    Args:
        x_train (np.ndarray): _description_

    Returns:
        int: index of the categorical variable in the feature matrix
    """
    for col in range(x_train.shape[D_DIM]):
        if len(np.unique(x_train[:, col])) == 4:
            return col


def categorical_subset_generator(
    y_train: np.ndarray,
    tx_train: np.ndarray,
    y_test: np.ndarray,
    tx_test: np.ndarray,
    categorical_groups: List[List[int]],
    categorical_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        y_train (np.ndarray): train labels.
        tx_train (np.ndarray): train samples.
        y_test (np.ndarray): test labels.
        tx_test (np.ndarray): test samples
        categorical_groups (List[List[int]]): groups of values for categorical
        variable.
        categorical_idx (int): index of the categorical feature

    Yields:
        Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: it
        yields the train samples and labels, and test samples and labels for
        for each of the generated subsets.
    """

    # If it is only one group, we don't bother with all the logic (faster iteration)
    if len(categorical_groups) == 1:
        yield y_train, tx_train, y_test, tx_test
        return

    N_train = y_train.shape
    N_test = y_test.shape

    train_categorical_column = tx_train[:, categorical_idx]
    test_categorical_column = tx_test[:, categorical_idx]

    for categorical_group in categorical_groups:

        train_categorical_mask = np.zeros((N_train))
        test_categorical_mask = np.zeros((N_test))

        for categorical_value in categorical_group:
            new_train_mask = train_categorical_column == categorical_value
            train_categorical_mask = np.logical_or(
                train_categorical_mask, new_train_mask
            )
            new_test_mask = test_categorical_column == categorical_value
            test_categorical_mask = np.logical_or(test_categorical_mask, new_test_mask)

        y_train_subset = y_train[train_categorical_mask]
        tx_train_subset = tx_train[train_categorical_mask]
        y_test_subset = y_test[test_categorical_mask]
        tx_test_subset = tx_test[test_categorical_mask]

        # Remove the categorical column from the subset
        tx_train_subset = np.delete(tx_train_subset, categorical_idx, axis=1)
        tx_test_subset = np.delete(tx_test_subset, categorical_idx, axis=1)

        yield y_train_subset, tx_train_subset, y_test_subset, tx_test_subset
