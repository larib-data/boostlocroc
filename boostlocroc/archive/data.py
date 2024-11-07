"""Data processing."""

import glob
import os.path as op
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def get_input_target(data):
    """Get input and target from the psd's labels csv."""
    df = data.copy()
    X = df.iloc[:, :-1].set_index("ID")
    Y = df.iloc[:, [0, -1]].set_index("ID").squeeze()

    return X, Y


def sample_df(df):
    """Get input and target from the psd's labels csv."""
    df.drop(["epochs"], axis=1, inplace=True)

    X = df.iloc[:, :-1].set_index("ID")
    Y = df.iloc[:, [0, -1]].set_index("ID").squeeze()

    return X, Y


def get_subject(X, Y, subject_name, labels):
    """Get the right subject."""

    subject_label = labels.loc[subject_name]
    y_true = (subject_label.LOC_time, subject_label.ROC_time)

    return X.loc[subject_name], Y.loc[subject_name], y_true


def get_paths_list(directory, regex):
    """Get a list of path."""
    return np.sort(glob.glob(op.join(directory, regex)))


def get_labels(csv_file, sort_by="ID", sep=";"):
    """Load the labels."""
    # Loading the labels for each patient and sort them by ID
    return pd.read_csv(csv_file, sep=sep).sort_values(sort_by).set_index(sort_by)


def get_lists_of_raw_files():
    """Get a lists of raw_files."""
    parquets_list = get_paths_list(data_src_dir, "*/*" + parquet_extension)
    raws_list = get_paths_list(data_src_dir, "*/*" + raw_extension)

    return parquets_list, raws_list


# Dataset builder
def split_dataset(X, Y, test_size=0.2, random_state=42):
    """Split the dataset."""
    # Split data in test/train subgroups according to a provided group
    groups = X.index

    train_inds, test_inds = next(
        GroupShuffleSplit(
            n_splits=2, test_size=test_size, random_state=random_state
        ).split(X, groups=groups)
    )
    X_train, X_test, y_train, y_test = (
        X.iloc[train_inds],
        X.iloc[test_inds],
        Y[train_inds],
        Y[test_inds],
    )

    train = (X_train, y_train)
    test = (X_test, y_test)

    return train, test


def subsample_dataset(X, Y, batching_size_factor=2, random_state=42,
                      binary=True):
    """Get input and target from the psd's labels csv with batching a priori.
    """

    train_set = pd.concat([X, Y], axis=1)
    train_1 = train_set[train_set.labels == 1]

    if binary:
        train_0 = train_set[train_set.labels == 0]
    else:
        train_0 = train_set[(train_set.labels == 0) | (train_set.labels == 2)]

    sample_size = int(
        batching_size_factor * len(train_0) / len(train_set.index.unique())
    )

    train_sample = train_1.groupby("ID").sample(
        sample_size, random_state=random_state, replace=True
    )
    train_batch = pd.concat([train_sample, train_0], axis=0).sort_values(
        by=["ID", "epochs"]
    )

    train_batch.drop(["epochs"], axis=1, inplace=True)

    X = train_batch.iloc[:, :-1]  # .set_index('ID')
    Y = train_batch.iloc[:, -1]  # [0, -1]].set_index('ID').squeeze()

    return X, Y


def weighted_dataset(X, Y, batching_size_factor=2, random_state=42,
                     binary=True):
    """Get input and target from the psd's labels csv with batching a priori.
    """

    train_set = pd.concat([X, Y], axis=1)
    train_1 = train_set[train_set.labels == 1]

    if binary:
        train_0 = train_set[train_set.labels == 0]
    else:
        train_0 = train_set[(train_set.labels == 0) | (train_set.labels == 2)]

    epochs_loc = train_1.groupby("ID").head(6)
    epochs_roc = train_1.groupby("ID").tail(12)
    epochs = (
        train_1.groupby("ID", as_index=False)
        .apply(lambda group: group.iloc[6:-12])
        .reset_index(level="ID")
        .set_index("ID")
    )
    sample_size = int(
        batching_size_factor * len(train_0) / len(train_set.index.unique())
    )
    train_sample = epochs.groupby("ID").sample(
        sample_size, random_state=random_state, replace=True
    )

    train_batch = pd.concat(
        [train_sample, train_0, epochs_loc, epochs_roc], axis=0
    ).sort_values(by=["ID", "epochs"])

    train_batch.drop(["epochs"], axis=1, inplace=True)

    X = train_batch.iloc[:, :-1]
    Y = train_batch.iloc[:, -1]

    return X, Y


def load_idx_train_test(template_name="cross_val_",
                        weights_dir="model_weights"):
    """Load index."""
    pickle_in = open(op.join(weights_dir, f"{template_name}_idx_spliting.pickle"), "rb")
    (idx_train, idx_test) = pickle.load(pickle_in)
    return idx_train, idx_test
