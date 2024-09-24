"""Prediction tools for LOC and ROC."""

from eeg_features import (
    smooth_probability,
    predict_gbc,
    __fun_LOC,
    __fun_ROC,
    __objective_ROC,
    __objective_LOC,
)
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os.path as op
from visualization import set_ax_setting, add_ax_vlines
from scipy.optimize import least_squares


def min_max_normalization(signal):
    """
    Normalize the signal between 0 and 1.

    Parameters
    ----------
     signal : :obj:`list` or :obj:`pandas.core.series.Series` or :obj:`numpy.ndarray`
         Signal to process.

    Returns
    -------
     signal : :obj:`list`
         The normalized signal.
    """
    min_signal = np.min(signal)
    diff_max_min_signal = np.max(signal) - min_signal

    signal = [(ele - min_signal) / diff_max_min_signal for ele in signal]

    return signal


def least_square_pred(
    t,
    probability,
    mode="loc",
    loss="soft_l1",
    f_scale=0.1,
    min_dur_intervention=30,
    time_loc=0,
):
    """Compute a least square prediction."""
    if mode == "loc":
        fun = __fun_LOC
        x0 = np.array([0.2, 15 * 60])
        condition = 240 * 30
        fun_args = (
            t[t < condition],  # 110
            probability[t < condition],
        )
        bounds = ([0.005, 0.5], [5, 1200 * 60])

    elif mode == "roc":
        fun = __fun_ROC
        x0 = np.array([0.2, t[int(len(t) * 0.95)]])
        fun_args = (
            t[t > (time_loc + min_dur_intervention * 60)],
            probability[t > (time_loc + min_dur_intervention * 60)],
        )
        bounds = ([0.005, 0.5], [(time_loc + min_dur_intervention * 60), max(t)])

    res_robust = least_squares(
        fun,
        x0,
        loss=loss,
        f_scale=f_scale,
        args=fun_args,
        bounds=bounds,
    )

    return res_robust.x[1]  # + (1 / res_robust.x[0]) * np.log(4.5)


def predict_without_sigmoid(X, weights, epochs_duration=30):
    """Prediction."""
    probability = predict_gbc(X, weights)
    probability = smooth_probability(probability)
    return probability


def predict_probabilities(model, X, min_max_norm=False):
    """Probabilities prediction."""
    probabilities = model.predict_proba(
        X,
    )
    probabilities = smooth_probability(probabilities)
    if "epochs" in X.columns:
        X.drop("epochs", inplace=True, axis=1)
    if min_max_norm:
        probabilities = np.array(min_max_normalization(probabilities))

    return probabilities


def predict_loc_roc(
    probability,
    epochs_duration=30,
    y_true=(0, 0),
):
    """Prediction LOC and ROC."""
    # probability = probability["probabilities"].to_numpy()
    L = len(probability)
    t = np.linspace(0, L * epochs_duration, L)
    min_dur_intervention = 0

    time_loc = least_square_pred(
        t, probability, min_dur_intervention=min_dur_intervention
    )
    time_roc = least_square_pred(
        t,
        probability,
        mode="roc",
        time_loc=time_loc,
        min_dur_intervention=min_dur_intervention,
    )

    return time_loc, time_roc
