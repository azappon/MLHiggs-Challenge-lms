import ast
import csv
import functools
import itertools
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple, Any, Union, NamedTuple
import numpy as np
import pickle
import os
from tqdm import tqdm

import dataset_utils
import linear_regression_utils
import logistic_regression_utils
import preprocessing
from implementations import (
    ridge_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    logistic_regression,
    logistic_regression_sgd,
    reg_logistic_regression_sgd,
    least_squares,
)

from naive_baselines import (
    get_random_prediction,
    majority_guess,
    get_majority_prediction,
    random_guess,
)

N_DIM = 0
D_DIM = 1


def preprocess_dataset(
    tx_train: np.ndarray,
    y_train: np.ndarray,
    tx_test: np.ndarray,
    categorical_idx: np.int,
    hyperparameters: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses the datasets.

    Args:
        tx_train (np.ndarray): training samples.
        y_train (np.ndarray): training labels.
        tx_test (np.ndarray): test samples.
        categorical_idx (np.int): index of the categorical features.
        hyperparameters (_type_): dictionary containing all the hyperparameters
        controlling the preprocessing steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: processed train and test samples, respectively.
    """

    if len(hyperparameters["categorical_subsets"]) == 1:
        # Only one subset, use one-hot encoding.
        tx_train = preprocessing.categorical_2_1hot(tx_train, [categorical_idx])
        tx_test = preprocessing.categorical_2_1hot(tx_test, [categorical_idx])

    tx_train, tx_test = preprocessing.handle_nans(
        tx_train,
        tx_test,
        threshold_percentage=hyperparameters["nan_threshold_ratio"],
        replacement_type=hyperparameters["nan_replacement_type"],
    )

    if hyperparameters["extrema_clip_percentage"] > 0:
        tx_train, tx_test = preprocessing.clip_features_outside_range(
            tx_train, tx_test, alpha=hyperparameters["extrema_clip_percentage"]
        )
    if hyperparameters["non_informative_threshold"] > 0:
        tx_train, tx_test = preprocessing.remove_non_informative_features(
            tx_train,
            y_train,
            tx_test,
            threshold=hyperparameters["non_informative_threshold"],
        )

    # Standardize before feature augmentation to avoid overflows.
    if hyperparameters["standardize"]:
        tx_train, tx_test = preprocessing.standardize(tx_train, tx_test)

    if hyperparameters["feature_augmentation"]:
        tx_train, tx_test = preprocessing.feature_augmentation(
            tx_train, tx_test, hyperparameters
        )

    # Restandardize. Only applies to the new features and is idempotent for already normalized features.
    if hyperparameters["standardize"]:
        tx_train, tx_test = preprocessing.standardize(tx_train, tx_test)

    tx_train = preprocessing.add_intercept(tx_train)
    tx_test = preprocessing.add_intercept(tx_test)

    return tx_train, tx_test


def k_fold_cross_validation(
    y: np.ndarray,
    tx: np.ndarray,
    algorithm: Callable,
    hyperparameters: Dict[str, Any],
    seed=77,
    k_fold=5,
) -> List[float]:
    """K-fold cross-validation implementation.

    Args:
        y (np.ndarray): dataset labels.
        tx (np.ndarray): dataset features.
        algorithm (Callable): algorithm to run in k-fold cross validation.
        hyperparameters (Dict[str, Any]): dictionary containing the training hyperparameters.
        seed (int, optional): random seed. Defaults to 77.
        k_fold (int, optional): number of folds applied in k-fold cross validation.
        cross-validation. Defaults to 5.

    Returns:
        List[float]: list of the test accuracies obtained in the different folds.
    """

    # Get the index of the categorical variable
    categorical_idx = dataset_utils.get_categorical_idx(tx)

    algorithm_accuracies = []
    for y_train, tx_train, y_test, tx_test in k_fold_generator(y, tx, seed, k_fold):

        fold_test_accuracy = 0
        total_size = 0

        # Iterate over each of the subsets (each subset corresponds to a set of
        # values of the categorical value). Those sets are prescribed by
        # hyperparameters["categorical_subsets"].
        for (
            y_train_subset,
            tx_train_subset,
            y_test_subset,
            tx_test_subset,
        ) in dataset_utils.categorical_subset_generator(
            y_train,
            tx_train,
            y_test,
            tx_test,
            hyperparameters["categorical_subsets"],
            categorical_idx,
        ):

            tx_train_subset, tx_test_subset = preprocess_dataset(
                tx_train_subset,
                y_train_subset,
                tx_test_subset,
                categorical_idx,
                hyperparameters,
            )

            # Getting a train and test routine to run on the preprocessed
            # k-fold datasets
            train_routine, test_routine = get_algorithm_routines(
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                num_features=tx_train_subset.shape[D_DIM],
            )
            w, _ = train_routine(y_train_subset, tx_train_subset)

            fold_subset_test_accuracy = test_routine(y_test_subset, tx_test_subset, w)

            # Weight the subset test accuracies by the number of samples in the
            # given subset
            total_size += len(y_test_subset)
            fold_test_accuracy += fold_subset_test_accuracy * len(y_test_subset)

        fold_test_accuracy /= total_size
        algorithm_accuracies.append(fold_test_accuracy)

    return algorithm_accuracies


def k_fold_generator(
    y: np.ndarray,
    tx: np.ndarray,
    seed: int,
    k_fold: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K-fold generator. Used to obtain the several train and test sets.

    Args:
        y (np.ndarray): dataset labels.
        tx (np.ndarray): dataset features.
        seed (int): random seed.
        k_fold (int): number of dataset folds.

    Yields:
        Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: tuple
        containing y_train, tx_train, y_test, tx_test for each dataset fold.
    """

    N = len(y)
    largest_divisible_N = int(k_fold * np.floor(N / k_fold))

    # Folds indices organized per rows
    # (indices in the same row belong to the same fold)
    rng = np.random.default_rng(seed)
    randomized_idxs = rng.permutation(np.arange(N))
    regular_idxs = randomized_idxs[:largest_divisible_N].reshape((k_fold, -1))

    # If elements can not be allocated evenly, they are stored in this array
    extra_idxs = randomized_idxs[largest_divisible_N:]

    # Iterate over folds
    k_idxs = np.arange(k_fold)
    for k_idx in k_idxs:

        # Get indices from regular matrix
        regular_test_idxs = regular_idxs[k_idx, :].flatten()
        regular_train_idxs = regular_idxs[k_idxs != k_idx, :].flatten()

        # Allocate extra indices to the corresponding folds
        if k_idx < len(extra_idxs):
            extra_test_idx = extra_idxs[k_idx]
            extra_train_idxs = np.delete(extra_idxs, k_idx)
        else:
            extra_test_idx = np.ndarray([], dtype=int)
            extra_train_idxs = extra_idxs

        # Get definitive indices
        test_idxs = np.hstack((regular_test_idxs, extra_test_idx))
        train_idxs = np.hstack((regular_train_idxs, extra_train_idxs))

        # Get test samples
        y_test = y[test_idxs]
        tx_test = tx[test_idxs]

        # Get train samples
        y_train = y[train_idxs]
        tx_train = tx[train_idxs]

        yield y_train, tx_train, y_test, tx_test


def get_algorithm_hyperparameters(algorithm: Callable) -> List[str]:
    """Identifies the hyperparameters for each algorithm.

    Args:
        algorithm (Callable): algorithm to which we get the hyperparameters.

    Returns:
        List[str]: list of all the hyperparameters for the provided algorithm.
    """

    if algorithm in [least_squares, random_guess, majority_guess]:
        algorithm_hyperparameters = []
    elif algorithm in [ridge_regression]:
        algorithm_hyperparameters = ["lambda_"]
    elif algorithm in [
        mean_squared_error_gd,
        mean_squared_error_sgd,
        logistic_regression,
        logistic_regression_sgd,
    ]:
        algorithm_hyperparameters = ["gamma", "max_iters"]
    elif algorithm in [
        reg_logistic_regression,
        reg_logistic_regression_sgd,
    ]:
        algorithm_hyperparameters = ["lambda_", "gamma", "max_iters"]
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented")

    return algorithm_hyperparameters


def hyperparameter_sweep(
    y: np.ndarray,
    tx: np.ndarray,
    config: SimpleNamespace,
) -> list[dict[Union[str, Any], Any]]:
    """Train the algorithm algorithm on the dataset (tx, y) on all combinations
    of hyperparameters in hyperparameters_domain with cross-validation and
    return the accuracies obtained in the cross-validation.

    Args:
        y (np.ndarray): dataset labels,
        tx (np.ndarray): dataset samples.
        config (SimpleNamespace): configurations file for the hyperparameter
        sweep.

    Returns:
        list[dict[Union[str, Any], Any]]: list of dictionaries, each containing
        a combination of hyperparameters and the corresponding test accuracies
    """

    # Define arguments for the provided algorithm, according to its signature.
    algorithm_hyperparameters = get_algorithm_hyperparameters(config.algorithm)

    # Create grid to iterate over.
    eligible_hyperparameters = {
        k: v
        for k, v in config.algorithm_hyperparams.items()
        if k in algorithm_hyperparameters
    }
    eligible_hyperparameters.update(config.preconditioning_hyperparams)
    eligible_hyperparameters.update(config.feature_augmentation_hyperparams)
    eligible_hyperparameters["seed"] = [config.seed]
    hyperparameter_grid = get_hyperparameter_combinations(eligible_hyperparameters)

    results = []
    for hyperparameter_comb in tqdm(hyperparameter_grid):
        # K-Fold cross-validation.
        test_accuracy_list = k_fold_cross_validation(
            y=y,
            tx=tx,
            algorithm=config.algorithm,
            hyperparameters=hyperparameter_comb,
            seed=config.seed,
            k_fold=config.k_fold,
        )

        # Store result in a structured way
        combination_data = generate_results_entry(
            hyperparameter_comb=hyperparameter_comb,
            test_accuracy_list=test_accuracy_list,
        )
        results.append(combination_data)

    return results


def get_hyperparameter_combinations(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Combines the items of a dictionary to form a "grid".

    Adapted from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    Example:
    dict_product(dict(number=[1,2], character='ab'))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]

    Args:
        d (Callable): dictionary from whose items we create grid combination.

    Returns:
        List[Dict[str, Any]]: list of grid composing dictionaries.
    """

    return [dict(zip(d.keys(), values)) for values in itertools.product(*d.values())]


def compute_categorical_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> np.float_:
    """Compute categorical accuracy between predictions and groundtruth.

    Args:
        y_pred (np.ndarray): labels predicted.
        y_true (np.ndarray): groundtruth labels.

    Returns:
        np.float_: categorical accuracy of the predictions.
    """
    return np.mean(y_pred == y_true).squeeze()


def get_algorithm_predictor(algorithm):
    if algorithm in [
        mean_squared_error_gd,
        mean_squared_error_sgd,
        least_squares,
        ridge_regression,
    ]:
        return linear_regression_utils.get_prediction
    elif algorithm in [
        logistic_regression,
        logistic_regression_sgd,
        reg_logistic_regression,
        reg_logistic_regression_sgd,
    ]:
        return logistic_regression_utils.get_prediction
    elif algorithm in [random_guess]:
        return get_random_prediction
    elif algorithm in [majority_guess]:
        return get_majority_prediction
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented.")


def get_algorithm_routines(
    algorithm: Callable, hyperparameters: dict[str, List], num_features: int
) -> Tuple[Callable, Callable]:
    """Get training and test routines for a given algorithm.

    Args:
        algorithm (Callable): algorithm to which we create the training and test
        routines.
        hyperparameters (dict[str, List]): hyperparameters considered in the
        routines generated.
        num_features (int): number of features, which corresponds to the number
        of parameters of the model.

    Returns:
        Tuple[Callable, Callable]: training and test routines.
    """

    initial_w = np.zeros(num_features)
    if algorithm == mean_squared_error_gd:
        train_routine = functools.partial(
            mean_squared_error_gd,
            initial_w=initial_w,
            max_iters=hyperparameters["max_iters"],
            gamma=hyperparameters["gamma"],
        )
    elif algorithm == mean_squared_error_sgd:
        train_routine = functools.partial(
            mean_squared_error_sgd,
            initial_w=initial_w,
            max_iters=hyperparameters["max_iters"],
            gamma=hyperparameters["gamma"],
        )
    elif algorithm == least_squares:
        train_routine = least_squares
    elif algorithm == ridge_regression:
        train_routine = functools.partial(
            ridge_regression, lambda_=hyperparameters["lambda_"]
        )
    elif algorithm == logistic_regression:
        train_routine = functools.partial(
            logistic_regression,
            initial_w=initial_w,
            max_iters=hyperparameters["max_iters"],
            gamma=hyperparameters["gamma"],
        )
    elif algorithm == logistic_regression_sgd:
        train_routine = functools.partial(
            logistic_regression_sgd,
            initial_w=initial_w,
            max_iters=hyperparameters["max_iters"],
            gamma=hyperparameters["gamma"],
        )
    elif algorithm == reg_logistic_regression:
        train_routine = functools.partial(
            reg_logistic_regression,
            initial_w=initial_w,
            max_iters=hyperparameters["max_iters"],
            gamma=hyperparameters["gamma"],
            lambda_=hyperparameters["lambda_"],
        )
    elif algorithm == reg_logistic_regression_sgd:
        train_routine = functools.partial(
            reg_logistic_regression_sgd,
            initial_w=initial_w,
            max_iters=hyperparameters["max_iters"],
            gamma=hyperparameters["gamma"],
            lambda_=hyperparameters["lambda_"],
        )
    elif algorithm == random_guess:
        train_routine = functools.partial(random_guess, seed=hyperparameters["seed"])
    elif algorithm == majority_guess:
        train_routine = majority_guess
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented.")
    algorithm_predictor = get_algorithm_predictor(algorithm)
    test_routine = lambda y_test, tx_test, w: compute_categorical_accuracy(
        algorithm_predictor(tx_test, w), y_test
    )
    return train_routine, test_routine


def generate_results_entry(
    hyperparameter_comb: Dict[str, float], test_accuracy_list: List[float]
) -> Dict[str, Union[float, List[float]]]:
    """Creates a dictionary containing a given combination of hyperparameters
    and the corresponding test accuracies.

    Args:
        hyperparameter_comb (Dict[str, float]): combination of hyperparameters.
        test_accuracy_list (List[float]): list of test accuracies for the given
        hyperparameter combination.

    Returns:
        Dict[str, Union[float, List[float]]]: dictionary containing the
        hyperparameter prescription and list of test accuracies.
    """
    entry_data = {}
    entry_data.update(hyperparameter_comb)
    entry_data["test_accuracies"] = test_accuracy_list

    return entry_data


def save_hyperparameter_sweep_results(
    algorithm: Callable,
    results: List[Dict[str, Union[float, List[float]]]],
    output_dir: str,
) -> None:
    """Saves hyperparameters sweep results.

    Args:
        algorithm (Callable): algorithm ran in the hyperparameter sweep.
        results (List[Dict[str, Union[float, List[float]]]]): list containing
        dictionaries, each with the result of one hyperparameter combination.
        output_dir (str): directory to save.
    """
    csv_path = f"{output_dir}/{algorithm.__name__}.csv"
    pickle_path = f"{output_dir}/{algorithm.__name__}.pkl"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # CSV
    csv_header = list(results[0].keys())
    with open(csv_path, "w") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=csv_header, quoting=csv.QUOTE_NONNUMERIC
        )
        writer.writeheader()
        for combination_data in results:
            writer.writerow(combination_data)

    # PICKLE
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(results, pickle_file)


def load_hyperparameter_sweep_results(
    file_path: str,
) -> List[Dict[str, Union[float, List[float]]]]:
    """Loads hyperparameter results from file. Supports .csv and .pkl

    Args:
        file_path (str): path to the results storing file.

    Returns:
        List[Dict[str, Union[float, List[float]]]]: list containing
        dictionaries, each with the result of one hyperparameter combination.
    """
    # CSV (broken by now, please use pickle)
    if file_path[-4:] == ".csv":
        # Read csv
        with open(file_path, "r") as file_read:
            results_list = list(csv.DictReader(file_read, quoting=csv.QUOTE_NONNUMERIC))

        # Test accuracies are stored as a string.
        # We have to convert them back to a list of floats
        for entry in results_list:
            entry["test_accuracies"] = ast.literal_eval(entry["test_accuracies"])

    # Pickle
    elif file_path[-4:] == ".pkl":
        with open(file_path, "rb") as file_read:
            results_list = pickle.load(file_read)

    else:
        NotImplementedError("It is not possible to read such format.")

    return results_list


def pick_best_hyperparameters(
    results: List[Dict[str, Union[float, List[float]]]]
) -> Tuple[Dict[str, Union[float, List[float]]], float]:
    """Picks best hyperparameter combination from all the combinations run.

    Args:
        results (List[Dict[str, Union[float, List[float]]]]): list containing
        dictionaries, each with the result of one hyperparameter combination.

    Returns:
        Tuple[Dict[str, Union[float, List[float]]], float]: dictionary
        containing the best hyperparameter combination and respective test
        accuracies, and the mean accuracy for that same hyperparameter
        combination.
    """

    best_mean_accuracy = 0
    best_hyperparameters = None
    for entry in results:
        if best_mean_accuracy < np.mean(entry["test_accuracies"]):
            best_mean_accuracy = np.mean(entry["test_accuracies"])
            best_hyperparameters = entry

    return best_hyperparameters, best_mean_accuracy


def get_accuracies_statistics(test_accuracies: List[float]) -> Tuple[float, float]:
    """Gets the mean and standard deviation of a list of test accuracies.

    Args:
        test_accuracies (List[float]): list of test accuracies.

    Returns:
        Tuple[float, float]: mean and standard deviation of the list of
        accuracies.
    """
    return np.mean(test_accuracies), np.std(test_accuracies)
