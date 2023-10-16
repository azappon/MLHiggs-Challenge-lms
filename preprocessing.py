from typing import Any, Dict, List, NamedTuple, Tuple, Union
import numpy as np
from collections import namedtuple


def categorical_2_1hot(x: np.ndarray, categorical_idxs: List[int]) -> np.ndarray:
    """Convert categorical features into their one-hot representation.

    Categorical columns are assumed to be already in categorical format, i.e.
    they only take values in {0, 1, ..., K}.

    Args:
        x (np.ndarray):
        categorical_idxs (List[int]): list of categorical features indices.

    Returns:
        np.ndarray: new feature matrix with the categorical variable converted to
        its one-hot representation. The categorical features are deleted and
        their one-hot representations are added in the end (on the right) of
        the feature matrix.
    """
    # Categorical columns already assumed in categorical format, i.e. they only take values in {0, 1, ..., K}

    N, d = x.shape
    output_x = x.copy()

    for categorical_idx in categorical_idxs:
        # Get categorical column
        categorical_column = x[:, categorical_idx].astype(int)
        # Create one hot encoded representation of that column
        one_hot_encoded_columns = np.zeros((N, categorical_column.max() + 1))
        one_hot_encoded_columns[np.arange(N), categorical_column] = 1
        # Append one hot encoded columns
        output_x = np.hstack((output_x, one_hot_encoded_columns))

    # Delete original categorical variables
    categorical_mask = np.zeros((output_x.shape[1],), dtype=bool)
    categorical_mask[categorical_idxs] = True
    output_x = output_x[:, ~categorical_mask]

    return output_x


def standardize(tx_train, tx_test):
    """Standardize the training and testing data.

    Args:
        tx_train (np.ndarray): training features. Shape: (N_train, d)
        tx_test (np.ndarray): testing features. Shape: (N_test, d)
    Returns:
        tx_train (np.ndarray): standardized training features. Shape: (N_train, d)
        tx_test (np.ndarray): standardized testing features. Shape: (N_test, d)
    """
    train_mean = np.mean(tx_train, axis=0)
    train_std = np.std(tx_train, axis=0)
    tx_train = (tx_train - train_mean) / train_std
    tx_test = (tx_test - train_mean) / train_std

    return tx_train, tx_test


def handle_nans(
    x_train: np.ndarray,
    x_test: np.ndarray,
    threshold_percentage: float = 0.8,
    replacement_type: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Deletes features whose features are -999 if their fraction is higher than
    threshold percentage. Otherwise, replaces those values with the feature
    mean or median

    Args:
        x_train (np.ndarray): dataset samples.
        x_test (np.ndarray): dataset samples.
        threshold_percentage (float, optional): threshold above which the
        feature is deleted. Defaults to 0.8.
        replacement_type (str, optional): statistical representative to replace
        -999 values if their feature fraction is smaller than
        threshold_percentage. Defaults to "mean".

    Returns:
        Tuple[np.ndarray, np.ndarray]: cleaned train and test samples.
    """
    missing_values_fraction = np.mean(x_train == -999, axis=0)

    # Keep the features with a missing percentage less than threshold
    keep_mask = missing_values_fraction < threshold_percentage
    clean_x_train = x_train[:, keep_mask]
    clean_x_test = x_test[:, keep_mask]

    # For the remaining features, replace with a distribution representative
    # (mean or media)
    if replacement_type == "mean":
        statistic_operation = np.mean
    elif replacement_type == "median":
        statistic_operation = np.median
    else:
        raise NotImplementedError("The replacement type provided is not available.")

    _, d = clean_x_train.shape
    nan_mask_train = clean_x_train == -999
    nan_mask_test = clean_x_test == -999
    for feature_idx in range(d):
        nan_mask_feature_train = nan_mask_train[:, feature_idx]
        nan_mask_feature_test = nan_mask_test[:, feature_idx]
        statistic_feature = statistic_operation(
            clean_x_train[~nan_mask_feature_train, feature_idx]
        )
        clean_x_train[nan_mask_feature_train, feature_idx] = statistic_feature
        clean_x_test[nan_mask_feature_test, feature_idx] = statistic_feature

    return clean_x_train, clean_x_test


def inverse_hyperbolic_sine(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Computes the inverse hyperbolic sine transformation of the features:
       IHS(x) = ln(x + \sqrt(x^2 + 1))

    Args:
        x (Union[float, np.ndarray]): data to apply the transformation.

    Returns:
        Union[float, np.ndarray]: transformed data
    """

    return np.log(x + np.sqrt(x**2 + 1))


def get_product_features(x: np.ndarray) -> np.ndarray:
    """Adds new features to the data by computing the product of each couple of features.

    Args:
        x (np.ndarray): data to apply the transformation.

    Returns:
        np.ndarray: product features from x.
    """

    # Select the indices of the features to multiply
    indices = np.triu_indices(x.shape[1], k=1)

    # Compute the product of the features
    x_product = np.multiply(x[:, indices[0]], x[:, indices[1]])

    return x_product


def clip_features_outside_range(
    x_train: np.ndarray, x_test: np.ndarray, alpha=1
) -> np.ndarray:
    """Clips the features outside of an alpha percentile in the training and testing dataset
    based on the training dataset.

    Args:
        x_train (np.ndarray): training samples.
        x_test (np.ndarray): test samples.
        alpha (int, optional): percentile. Defaults to 1.

    Returns:
        np.ndarray: The data with clipped outliers.
    """

    _, D = x_train.shape

    for i in range(D):
        alpha_percentile = np.percentile(x_train[:, i], alpha)
        beta_percentile = np.percentile(x_train[:, i], (100 - alpha))

        x_train[:, i][x_train[:, i] < alpha_percentile] = alpha_percentile
        x_train[:, i][x_train[:, i] > beta_percentile] = beta_percentile

        x_test[:, i][x_test[:, i] < alpha_percentile] = alpha_percentile
        x_test[:, i][x_test[:, i] > beta_percentile] = beta_percentile

    return x_train, x_test


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Adds a column of 1's to the data.

    Args:
        x (np.ndarray): data.

    Returns:
        np.ndarray: data with one column of ones concatenated at its right.
    """

    return np.c_[x, np.ones((x.shape[0], 1))]


def get_polynomial_features(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Computes the polynomial expansion of the features.

    Args:
        x (np.ndarray): data.
        max_degree (int): maximum degree to compute the polynomial expansion.

    Returns:
        np.ndarray: data with the polynomial expansion of the features from the given list.
    """

    polynomial_features = []
    for degree in range(2, max_degree + 1):
        polynomial_features.append(np.power(x, degree))

    return np.concatenate(polynomial_features, axis=1)


def get_non_linear_features(x: np.ndarray, hyperparams: Dict[str, any]):
    """
    Apply nonlinearities to the features of the dataset x.

    Args:
        x (np.ndarray): input data.
        hyperparams[non_linearity]
            ihs: if True, apply inverse hyperbolic sine
            sine: if True, apply sine to the features
            asinh: if True, apply asinh to the features
            cosine: if True, apply cosine to the features
            cosh: if True, apply cosh to the features
            exp: if True, apply exp to the features
            abs: if True, apply absolute value to the features

    Returns:
        x: input data with applied nonlinearities to the original features only
    """

    N, _ = x.shape
    non_linear_features = []
    if hyperparams["ihs"]:
        non_linear_features.append(inverse_hyperbolic_sine(x))
    if hyperparams["sine"]:
        non_linear_features.append(np.sin(x))
    if hyperparams["sinh"]:
        non_linear_features.append(np.sinh(x))
    if hyperparams["cosine"]:
        non_linear_features.append(np.cos(x))
    if hyperparams["cosh"]:
        non_linear_features.append(np.cosh(x))
    if hyperparams["exp"]:
        non_linear_features.append(np.exp(x))
    if hyperparams["abs"]:
        non_linear_features.append(np.abs(x))

    return np.concatenate(non_linear_features, axis=1)


def ks_2samp(signal: np.ndarray, background: np.ndarray) -> NamedTuple:
    """Compute the Kolmogorov-Smirnov statistic on 2 samples.

    Reference:
    [1] Hodges, J.L. Jr.,  "The Significance Probability of the Smirnov
        Two-Sample Test," Arkiv fiur Matematik, 3, No. 43 (1958).

    The formula used can be found at page 485 (5.3):
    http://archive.ymsc.tsinghua.edu.cn/pacm_download/116/6944-11512_2007_Article_BF02589501.pdf

    Args:
        signal (np.ndarray): signal sample
        background (np.ndarray): background sample

    Returns:
        NamedTuple: contains fields statistic and pvalue
    """

    signal = np.sort(signal)
    background = np.sort(background)

    signal_n = signal.shape[0]
    background_n = background.shape[0]
    if min(signal_n, background_n) == 0:
        raise ValueError("Data passed to ks_2samp cannot be empty!")

    data_all = np.concatenate([signal, background])
    # using searchsorted solves equal data problem
    cdf_signal = np.searchsorted(signal, data_all, side="right") / signal_n
    cdf_background = np.searchsorted(background, data_all, side="right") / background_n
    cddiffs = cdf_signal - cdf_background
    # Sign of minS must not be negative.
    minS = np.clip(-np.min(cddiffs), 0, 1)
    maxS = np.max(cddiffs)

    P = -np.inf

    n = signal_n
    m = background_n
    d = max(minS, maxS)

    z = np.sqrt((m * n) / (m + n)) * d

    # Requires background_n to be the larger of (signal_n, background_n)
    # Hodges' approximation (5.3)
    P = np.exp(-(2 * z**2) - ((2 / 3 * z) * ((m + 2 * n) / np.sqrt(m * n * (m + n)))))

    P = np.clip(P, 0, 1)

    KstestResult = namedtuple("KstestResult", ("statistic", "pvalue"))

    return KstestResult(d, P)


def compute_ks(tx: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computs the KS values of the distributions of signal an background

    Args:
        tx (np.ndarray): data
        y (np.ndarray): labels in {0, 1}

    Returns:
        np.ndarray: ks_values of the distributions of signal and background for each feature
    """

    initializer = np.zeros(tx.shape[1])
    for i in range(tx.shape[1]):
        # Compute the KS test for each feature
        ks = ks_2samp(tx[y == 1, i], tx[y == 0, i])
        initializer[i] = ks[0]
    ks_vector = initializer

    return ks_vector


def get_features_same_distr(ks_vector: np.ndarray, threshold: float) -> List[int]:
    """Gets the indices of features with the same distribution for different
    labels.

    Args:
        ks_vector (np.ndarray): KS_values of the distributions of signal and
        background for each feature
        threshold (float): threshold for the KS test

    Returns:
        List[int]: list of indices of the features with the same distribution
    """

    features_same_distr = []
    for i in range(len(ks_vector)):
        if ks_vector[i] < threshold:
            features_same_distr.append(i)
    return features_same_distr


def remove_non_informative_features(
    tx_train: np.ndarray,
    y_train: np.ndarray,
    tx_test: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Removes non-informative features using KS test.

    Args:
        tx_train (np.ndarray): train samples.
        y_train (np.ndarray): train labels.
        tx_test (np.ndarray): test samples
        threshold (float, optional): KS test threshold. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, np.ndarray]: clean train and test samples
    """

    ks_vector = compute_ks(tx_train, y_train)
    features_same_distr = get_features_same_distr(ks_vector, threshold)

    tx_train = np.delete(tx_train, features_same_distr, axis=1)
    tx_test = np.delete(tx_test, features_same_distr, axis=1)

    return tx_train, tx_test


def feature_augmentation(
    tx_train_subset: np.ndarray,
    tx_test_subset: np.ndarray,
    hyperparams: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs feature augmentation. In particulat, applies non-linear
    functions, products across all features and polynomial expansion

    Args:
        tx_train_subset (np.ndarray): train subset samples
        tx_test_subset (np.ndarray): test subset samples
        hyperparams (Dict[Any]): hyperparameters controlling the feature
        augmentation

    Returns:
        Tuple[np.ndarray, np.ndarray]: feature augmented train and test subset samples
    """

    # add non linearities
    train_non_linear_features = get_non_linear_features(tx_train_subset, hyperparams)
    test_non_linear_features = get_non_linear_features(tx_test_subset, hyperparams)

    # add products
    train_product_features = get_product_features(tx_train_subset)
    test_product_features = get_product_features(tx_test_subset)

    # add polynomials
    train_poly_features = get_polynomial_features(
        tx_train_subset, max_degree=hyperparams["poly_max_degree"]
    )
    test_poly_features = get_polynomial_features(
        tx_test_subset, max_degree=hyperparams["poly_max_degree"]
    )

    # concatenate all features
    feature_augmented_train = np.c_[
        tx_train_subset,
        train_non_linear_features,
        train_product_features,
        train_poly_features,
    ]
    feature_augmented_test = np.c_[
        tx_test_subset,
        test_non_linear_features,
        test_product_features,
        test_poly_features,
    ]

    return feature_augmented_train, feature_augmented_test
