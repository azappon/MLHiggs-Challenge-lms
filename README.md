# CS433 Project 1 - Team LMS

## Getting Started

### Installation

We support installing the project dependencies via `conda`,
which we recommend installing with [miniforge](https://github.com/conda-forge/miniforge).
We provide an `environment.yml` file to create a conda environment.
This file extends the environment file provided by the teaching team to run basic unit tests.

```bash
conda env create --file=environment.yml --name=cs433-p1-lms
conda activate cs433-p1-lms
```

### Basic Tests

We track the basic tests provided by the teaching team as
a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) of this project.
To pull the tests run

```bash
git submodule update --init
```

Note that you will also run this command after a `git pull` to merge the submodule changes as well.

To get a short summary of the tests, run

```bash
pytest --github_link . epfl-cs433/projects/project1/grading_tests -v --color=yes | head -n 36
```

Or for the full trace, run

```bash
pytest --github_link . epfl-cs433/projects/project1/grading_tests
```

### End-to-end Prediction

We provide a `run.py` script that generates our best predictions for the test set.
The script runs an end-to-end data loading and preprocessing, training, and inference pipeline
for the task with our best model configuration.
The predictions are saved in a `sumbission.csv` file in the root of the project.
This file corresponds to our submission to the competition platform.

First, extract the dataset

```bash
 unzip data/dataset.zip -d data
```

Then, run the prediction script

```bash
python run.py
```

## Repository Organization

The repository is organized as follows:

- `dataset_utils.py`: contains utilities that deal with the dataset;

- `environment.yml`: file containing the dependencies of this repository;

- `helpers.py`: (*provided by the teaching team*) functions to load the dataset and create a submission `.csv` file;

- `implementations.py`: implementations for the functions required in the project;

- `linear_regression_utils.py`: contains the utilities to run the several logistic regression method applied;

- `logistic_regression_utils.py`: has all the utilities to run the several logistic regression method applied;

- `naive_baselines.py`: has the naive algorithms implementations (random guess and majority guess);

- `plots.py`: has the plotting functions for the results of each model. In particular, it includes the boxplot function for the accuracies of different models, and also an errorplot for hyperparameter sweep;

- `preprocessing.py`: contains all the functions used to manipulate our datasets;

- `run.py`: produces exactly the same `.csv` predictions which we used in our final submission to the competition system;

- `train_utils.py`: has all the functions we use to train our models, including the hyperparameter search and corresponding data preprocessing, K-fold cross-validation, and I/O models and results;

- **Directories**:

  - `data` directory: must contain the dataset;

  - `notebooks` directory: notebooks created to obtain the results in the `results` and `figures` directories;

  - `results` directory: files containing the results obtained for the experiments presented in the report;

  - `figures` directory: figures obtained from running the notebooks in the `notebooks` directory (some of them in the report).

  - `epfl-cs433`: submodule from the class.


## Developer Guide

### Formatting

```bash
black --exclude epfl-cs433 .
```

### Git

#### Typical routines:

**Starting a new feature:**

- Update your main:
    - `git checkout main`
    - `git pull`
    - `git submodule update --init`
- Create your feature branch:
    - `git checkout -b your_name-your_feature`
- Do some changes.
- Format your code and check the tests.
- Commit your changes:
    - `git add <files>`
    - `git commit -m "Some meaningful and grammatically correct message."`

**Working on your feature:**

- Update your main:
    - `git checkout main`
    - `git pull`
    - `git submodule update --init`
- Checkout your feature branch:
    - `git checkout your_name-your_feature`
- Rebase from main:
    - `git rebase main`
- Do some changes.
- Format your code and check the tests.
- Commit your changes:
    - `git add <files>`
    - `git commit -m "Some meaningful and grammatically correct message."`

**Sharing your feature:**

- Update your main:
    - `git checkout main`
    - `git pull`
    - `git submodule update --init`
- Checkout your feature branch:
    - `git checkout your_name-your_feature`
- Rebase from main:
    - `git rebase main`
- Fast-forward merge to main (or create a pull request if you wish):
    - `git checkout main`
    - `git merge --ff your_name-your_feature`
- Push main:
    - `git push`