"""Module to define the GB and voting ensemble model."""
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import os.path as op
from joblib import load
import numpy as np


def get_models(template, n_splits):
    """Choose the model."""
    # Get the directory where the current script is located
    script_dir = op.dirname(op.abspath(__file__))
    # Define the relative path to the pkl file
    models = [load(op.join(script_dir, 'model_weights', f"{template}{i}.pkl")) for i in range(n_splits)]
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
    clf_list, y_train = None, voting="soft", n_jobs=-1, verbose=True
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

    script_dir = op.dirname(op.abspath(__file__))
    weight_dir = op.join(script_dir, 'model_weights', 'voting_classifier_weight.npy')
    
    if y_train is not None : 
        voting_classifier.le_ = LabelEncoder().fit(y_train)
        np.save(weight_dir, voting_classifier.le_.classes_)
        voting_classifier.classes_ = voting_classifier.le_.classes_

    else : 
        voting_classifier.le_ = LabelEncoder()
        voting_classifier.classes_ = np.load(weight_dir)
        voting_classifier.le_.classes_ = voting_classifier.classes_

    return voting_classifier
