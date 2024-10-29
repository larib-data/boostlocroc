"""Training module."""

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from joblib import dump
import os.path as op
import pickle


def train_cross_validation(
    X,
    Y,
    model,
    n_splits=3,
    random_state=42,
    shuffle=True,
    scoring=["balanced_accuracy", "roc_auc"],
    n_jobs=-1,
):
    """Compute a cross validation."""
    # KFold split
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    groups = X.index

    # Get KFold idx
    idx_train = []
    idx_val = []

    for train, test in cv.split(X, Y, groups=groups):
        idx_train.append(train)
        idx_val.append(test)
    # FIT
    cv_results = cross_validate(
        model,
        X,
        Y,
        scoring=scoring,
        groups=groups,
        cv=cv,
        n_jobs=n_jobs,
        return_estimator=True,
    )

    return cv_results, idx_train, idx_val


def train_and_save_cv(
    X,
    Y,
    model,
    weights_dir,
    n_splits=3,
    template_name="cross_val_",
    random_state=42,
):
    """Compute a cross validation."""
    # Train
    cv_results, idx_train, idx_val = train_cross_validation(
        X,
        Y,
        model,
        n_splits=n_splits,
        random_state=random_state,
    )

    # Save weights
    for i in range(n_splits):
        dump(
            cv_results["estimator"][i],
            op.join(weights_dir, f"{template_name}{i}.pkl"),
            compress=1,
        )

    pickle_out = open(op.join(weights_dir, f"{template_name}_cv_results.pickle"), "wb")
    pickle.dump(cv_results, pickle_out)
    pickle_out.close()

    pickle_out = open(
        op.join(weights_dir, f"{template_name}_idx_spliting.pickle"), "wb"
    )
    pickle.dump((idx_train, idx_val), pickle_out)
    pickle_out.close()

    return cv_results, idx_train, idx_val
