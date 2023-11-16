"""Create feature from PSD."""
import os.path as op
import pandas as pd
from tqdm.auto import tqdm
from mne.io import read_raw
from ..eeg_features import raw_segmentation, smooth_psd
from .data import get_labels


n_fft = 512  # hamming window duration (# time samples) 512=~8s at 63Hz
n_overlap = 128
num_features = 50
shift = 30
epochs_duration = 30  # epoch duration (s) 60
shift = 30  # The duration to separate events by (in seconds).


def __epochs_labels(subject_name, verbose=False):
    path_raw = op.join(data_src_dir, subject_name, subject_name + raw_extension)

    # Load raw
    raw = read_raw(path_raw, verbose=verbose)
    epochs = raw_segmentation(raw, epochs_duration, shift)

    return raw, epochs


def __loc_roc_from_subject_name(subject_name):
    # Load labels
    loc_roc_labels = get_labels(labels_csv, sep=",")
    subject = loc_roc_labels.loc[subject_name]

    return subject.LOC_time, subject.ROC_time


def __make_labels(epochs, loc, roc):
    labels_by_epoch = []

    labels = pd.DataFrame(smooth_psd(epochs, n_fft, n_overlap, num_features))
    for i in range(labels.shape[0]):
        if i * 30 < loc or i * 30 > roc:
            labels_by_epoch.append(0)
        else:
            labels_by_epoch.append(1)

    labels["labels"] = labels_by_epoch

    return labels


def __make_labels_3(epochs, loc, roc):
    """Make labels where : 0 = before loc, 1 = sleep, 2 = after roc."""
    labels_by_epoch = []

    labels = pd.DataFrame(smooth_psd(epochs, n_fft, n_overlap, num_features))
    print(labels.shape)
    for i in range(labels.shape[0]):
        if i * 30 < float(loc):
            labels_by_epoch.append(0)
        elif i * 30 > float(roc):
            labels_by_epoch.append(2)
        else:
            labels_by_epoch.append(1)

    labels["labels"] = labels_by_epoch

    return labels


def epochs_labels_dataset(dst_csv_file, binary=True):
    """Load labels."""
    loc_roc_labels = get_labels(labels_csv, sep=",")

    labels_by_epochs = {}

    if binary:
        for subject_name, row in tqdm(
            loc_roc_labels.iterrows(), total=loc_roc_labels.shape[0]
        ):

            loc, roc = __loc_roc_from_subject_name(subject_name)

            _, epochs = __epochs_labels(subject_name, loc, roc)

            labels_psd = __make_labels(epochs, loc, roc)
            labels_by_epochs[subject_name] = labels_psd

    else:
        for subject_name, row in tqdm(
            loc_roc_labels.iterrows(), total=loc_roc_labels.shape[0]
        ):

            loc, roc = __loc_roc_from_subject_name(subject_name)

            _, epochs = __epochs_labels(subject_name, loc, roc)

            labels_psd = __make_labels_3(epochs, loc, roc)
            labels_by_epochs[subject_name] = labels_psd

    labels_by_epochs = pd.concat(labels_by_epochs)
    labels_by_epochs.reset_index().rename(
        columns={"level_0": "ID", "level_1": "epochs"}
    ).to_csv(dst_csv_file, index=False)

    return labels_by_epochs
