"""Functions used to define the GB and voting ensemble scikit-learn models.
Archival: functions in this module are not used in the package. They are kept
for reference purposes."""

from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

import os.path as op
from joblib import load
import numpy as np


def get_models(weights_dir, template, n_splits):
    """Load scikit-learn models from disk.

    Should only be used if you want to train your own model.
    Only works with scikit-learn version 1.0.2.

    Parameters
    ----------
    weights_dir: str
        Path to the directory containing the models.
    template: str
        Template of the model names.
    n_splits: int
        Number of splits, also the number of models to load.

    Returns
    -------
    models: list
        List of loaded models.
    """
    models = [load(op.join(weights_dir, f"{template}{i}.pkl"))
              for i in range(n_splits)]
    return models


def create_voting_ensemble_model(
    clf_list, weights_dir, voting="soft", n_jobs=-1, verbose=True
):
    """Create a voting ensemble model from loaded sklearn models.

    Should only be used if you want to train your own model. Loaded
    models were GradientBoostingClassifier models.

    Parameters
    ----------
    clf_list: list
        List of loaded models.
    weights_dir: str
        Path to the directory containing the models.

    Returns
    -------
    voting_classifier: VotingClassifier
        Voting ensemble model.
    """
    voting_classifier = VotingClassifier(
        estimators=[
            ("gbc1", clf_list[0]),
            ("gbc2", clf_list[1]),
            ("gbc3", clf_list[2]),
        ],
        voting=voting,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    voting_classifier.estimators_ = clf_list
    voting_classifier.le_ = LabelEncoder()
    voting_classifier.classes_ = np.load(
        op.join(weights_dir, "voting_classifier_weight.npy")
    )
    voting_classifier.le_.classes_ = voting_classifier.classes_

    return voting_classifier


def define_option(subsampling, weighted, binary):
    res = "--"
    if subsampling:
        res += "s"
    if weighted:
        res += "w"
    if binary:
        res += "b"
    res += "--"
    return res


def load_voting_skmodel(
    n_splits=3,
    subsampling=False,  # subsampling option
    weighted=True,  # Weighted option
    binary=False,  # True = 2 labels / False = 3 labels
    seed=42,
):
    """ Loads pre-trained voting model from disk and returns it.

    Should only be used if you want to train your own model.

    Parameters
    ----------
    Options used for the model training.
    One model is available, with the following parameters:
    n_splits=3, subsampling=False, weighted=True, binary=False, seed=42.
    If you want to load a model with different options,
    you need to train it first.

    Returns
    -------
    voting_ensemble_model: sklearn model
        Voting ensemble model.
    """
    option = define_option(subsampling, weighted, binary)

    cross_val_pathname = f"cross_val_weights_{seed}_{option}"
    weights_dir = "boost_loc_roc/model_weights/"
    models = get_models(weights_dir, cross_val_pathname, n_splits)

    voting_ensemble_model = create_voting_ensemble_model(models, weights_dir)
    return voting_ensemble_model
