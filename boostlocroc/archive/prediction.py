"""Prediction tools for LOC and ROC."""
import numpy as np
from eeg_features import (
    __fun_LOC,
    __fun_ROC,
    predict_gbc,
    smooth_probability,
)
from scipy.optimize import least_squares


def min_max_normalization(signal):
    """
    Normalize the signal between 0 and 1.

    Parameters
    ----------
     signal : :obj:`list` or :obj:`pandas.core.series.Series`
      or :obj:`numpy.ndarray`
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
        bounds = ([0.005, 0.5],
                  [(time_loc + min_dur_intervention * 60), max(t)]
                  )

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
