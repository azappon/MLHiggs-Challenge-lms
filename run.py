"""
This script runs an end-to-end data loading and preprocessing, training, and inference pipeline
for the task with our best model configuration.
The results output by this script were used to generate our final submission to the competition system.
It outputs a csv file submission.csv in the root directory of this project with the predicted labels for the test data.
"""
from types import SimpleNamespace

import dataset_utils
import helpers
import implementations
import train_utils

N_DIM = 0
D_DIM = 1

config = SimpleNamespace(
    seed=270404,
    sub_sample=False,
    algorithm=implementations.ridge_regression,
    algorithm_hyperparams=dict(
        lambda_=[1e-7],
        max_iters=[1],
        gamma=[1.0],
    ),
    preconditioning_hyperparams=dict(
        categorical_subsets=[
            [[0], [1], [2, 3]],
        ],
        standardize=[True],
        nan_threshold_ratio=[0.5],
        nan_replacement_type=["median"],
        extrema_clip_percentage=[1],
        non_informative_threshold=[0.05],
    ),
    feature_augmentation_hyperparams=dict(
        feature_augmentation=[True],
        poly_max_degree=[12],
        ihs=[True],
        sine=[True],
        sinh=[True],
        cosine=[True],
        cosh=[True],
        exp=[True],
        abs=[True],
    ),
    k_fold=10,
)

# Data loading.
y_train, x_train, ids_train = helpers.load_csv_data(
    "data/train.csv", sub_sample=config.sub_sample
)
_, x_test, ids_test = helpers.load_csv_data(
    "data/test.csv", sub_sample=config.sub_sample
)

categorical_idx = dataset_utils.get_categorical_idx(x_train)
y_train = dataset_utils.from_11_to_01(y_train)

# Train a model.
results = train_utils.hyperparameter_sweep(y_train, x_train, config)
train_utils.save_hyperparameter_sweep_results(config.algorithm, results, "results/run")
# Pick the best hyperparameters.
best_hyperparams, best_mean_accuracy = train_utils.pick_best_hyperparameters(results)
print(
    f"Best sweep accuracy: {best_mean_accuracy}.\n"
    f"Obtained for hyperparameters: {best_hyperparams}."
)

# Train with the best hyperprameters
tx_train, tx_test = train_utils.preprocess_dataset(
    x_train, y_train, x_test, categorical_idx, best_hyperparams
)
train_routine, _ = train_utils.get_algorithm_routines(
    algorithm=config.algorithm,
    hyperparameters=best_hyperparams,
    num_features=tx_train.shape[D_DIM],
)
w, _ = train_routine(y_train, tx_train)

# Create submission.
y_test = train_utils.get_algorithm_predictor(config.algorithm)(tx_test, w)

# transform outputs to -1, 1
y_test = dataset_utils.from_01_to_11(y_test)
# save submission
helpers.create_csv_submission(ids_test, y_test, "submission.csv")
