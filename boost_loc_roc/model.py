"""Module to define the GB and voting ensemble model."""

from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import os.path as op
from joblib import load
import numpy as np


def get_models(weights_dir, template, n_splits):
    """Choose the model."""
    models = [load(op.join(weights_dir, f"{template}{i}.pkl")) for i in range(n_splits)]
    return models


def create_model(
    n_estimators=138,
    learning_rate=0.24210526315789474,
    min_samples_leaf=6,
    max_depth=2,
    min_samples_split=2,
    random_state=42,
    max_features=None,
):
    """Create a GB model."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        max_features=max_features,
    )
    return model


def create_voting_ensemble_model(
    clf_list, weights_dir, y_train=None, voting="soft", n_jobs=-1, verbose=True
):
    """Create a voting ensemble model."""
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

    if y_train is not None:
        voting_classifier.le_ = LabelEncoder().fit(y_train)
        np.save(
            op.join(weights_dir, "voting_classifier_weight.npy"),
            voting_classifier.le_.classes_,
        )
        voting_classifier.classes_ = voting_classifier.le_.classes_

    else:
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


