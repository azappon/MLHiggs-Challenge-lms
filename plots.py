import os
from typing import Any, Dict, List
import matplotlib
import matplotlib.pyplot as plt

from train_utils import (
    get_accuracies_statistics,
    pick_best_hyperparameters,
    load_hyperparameter_sweep_results,
)
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


def plot_sweep_over_hyperparam(
    file_path: str,
    swept_hyperparameter: str,
    fixed_hyperparameters: Dict[str, Any],
    output_path: str,
    y_range: List[float] = [0, 1],
    x_scale: str = "log",
) -> None:
    """Plots a sweep over a given hyperparameter with an error plot.

    Args:
        file_path (str): path of the file containing the results to be plotted.
        swept_hyperparameter (str): hyperparameter in the x-axis.
        fixed_hyperparameters (Dict[str, Any]): hyperparameters that are fixed.
        output_path (str): path to save the file.
        y_range (List[float], optional): limits of the y-axis. Defaults to
        [0, 1].
        x_scale (str, optional): scale of the x-axis. Defaults to "log".
    """

    def has_same_fixed_hyperparameters(
        fixed_hyperparameters: Dict[str, Any], hyperparameter_comb: Dict[str, Any]
    ) -> bool:
        """Checks if the hyperparams of both dictionaries are the same.

        Args:
            fixed_hyperparameters (Dict[str, Any]): hyperparameters to be fixed.
            hyperparameter_comb (Dict[str, Any]): hyperparameter combination.

        Returns:
            bool: True if hyperparameters combination is the same as the
            combination fixed hyperparameters.
        """

        same_hyperparams = True
        for key in fixed_hyperparameters.keys():
            if fixed_hyperparameters[key] != hyperparameter_comb[key]:
                same_hyperparams = False
                break
        return same_hyperparams

    def create_error_plot() -> matplotlib.figure:
        """Outputs a matplotlib figure with an error plot.

        Returns:
            matplotlib.figure: figure with an error plot.
        """

        fig, ax = plt.subplots()
        ax.errorbar(
            x=lambdas_list,
            y=mean_accuracies_list,
            yerr=std_accuracies_list,
            fmt="ko",
            ecolor="k",
            elinewidth=2,
            capsize=5,
        )
        ax.set_xlabel(swept_hyperparameter)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(y_range)
        ax.set_xscale(x_scale)

        return fig

    algorithm_results = load_hyperparameter_sweep_results(file_path=file_path)

    # Get data to plot
    lambdas_list = []
    mean_accuracies_list = []
    std_accuracies_list = []
    for hyperparameter_comb in algorithm_results:

        if has_same_fixed_hyperparameters(fixed_hyperparameters, hyperparameter_comb):
            lambdas_list.append(hyperparameter_comb[swept_hyperparameter])
            mean, std = get_accuracies_statistics(
                test_accuracies=hyperparameter_comb["test_accuracies"]
            )
            mean_accuracies_list.append(mean)
            std_accuracies_list.append(std)

    # Plot
    fig = create_error_plot()

    # Create directory if it does not exist and save figure
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path)


def plot_models_accuracy(
    paths_to_models: List[str],
    output_path: str,
    y_range: List[float] = [0, 1],
    x_label_rotation: float = 0,
    from_format: str = ".pkl",
) -> None:
    """Plot models accuracy with a boxplot.

    Args:
        paths_to_models (List[str]): list of paths to the results of the different models.
        output_path (str): path where the final figure is stored.
        y_range (List[float], optional): limits of the y-axis. Defaults to [0, 1].
        x_label_rotation (float, optional): rotation on the x-axis labels. Defaults to 0.
        from_format (str, optional): from which stored format the results should be read. Defaults to ".pkl".
    """

    def create_box_plot() -> matplotlib.figure:
        """Boxplot with the different model accuracies.

        Returns:
            matplotlib.figure: Boxplot with model accuracies.
        """
        fig, ax = plt.subplots()
        ax.boxplot(accuracies_list)
        x_legends = [
            algorithm_file[:-4] for algorithm_file in algo_names_list
        ]  # delete '.csv'
        ax.set_xticklabels(x_legends, fontsize=8, rotation=x_label_rotation)
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(y_range)
        plt.tight_layout()

        return fig

    # Get accuracies for best combination of hyperparameters
    accuracies_dict = {}
    for file_path in paths_to_models:
        file_results = load_hyperparameter_sweep_results(file_path=file_path)
        best_hyperparameters, best_mean_accuracy = pick_best_hyperparameters(
            results=file_results
        )
        accuracies_dict[best_mean_accuracy] = {
            "test_accuracies": best_hyperparameters["test_accuracies"],
            "algorithm": os.path.basename(file_path),
        }

    accuracies_list = []
    algo_names_list = []
    for mean_accuracty in sorted(accuracies_dict):
        accuracies_list.append(accuracies_dict[mean_accuracty]["test_accuracies"])
        algo_names_list.append(accuracies_dict[mean_accuracty]["algorithm"])

    # Boxplot
    fig = create_box_plot()

    # Create directory if it does not exist and save figure
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path)
