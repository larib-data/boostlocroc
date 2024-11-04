"""Core functions of boost-loc-roc

Functions
---------
    fun_LOC,
    fun_ROC,
    objective_LOC,
    objective_ROC,
    proba_to_tloc_troc: Convert probability to time of LoC and RoC.
    extract_loc_roc: Extract the LOC and ROC times from raw EEG data.
    extract_loc_roc_sklearn: Extract the LOC and ROC from raw EEG data, using
    the original sklearn model.
"""
import numpy as np
import mne
from scipy.optimize import least_squares
from boost_loc_roc.archive.model import load_voting_skmodel
from boost_loc_roc.eeg_features import smooth_probability, compute_input_sample
from boost_loc_roc.utils.onnx import onx_make_session, onx_predict_proba


def fun_LOC(x, t, y):
    """Minimization of the logistic regression."""
    return objective_LOC(t, x[0], x[1]) - y


def fun_ROC(x, t, y):
    """Minimization of the inverse of the logistic regression."""
    return objective_ROC(t, x[0], x[1]) - y


def objective_LOC(x, a, x0):
    """Logistic regression."""
    b = -a * (x - x0)
    b = np.where(b > 500, 500, b)
    return (1 + np.exp(b)) ** (-1)


def objective_ROC(x, a, x0):
    """1 - Logistic regression."""
    b = -a * (x - x0)
    b = np.where(b > 500, 500, b)
    return 1 - (1 + np.exp(b)) ** (-1)


def proba_to_tloc_troc(probability, epochs_duration):
    """Convert probability to time of LoC and RoC.

    Parameters
    ----------
    probability : np.ndarray, shape (n_epochs,)
        Probability of each epoch.
    epochs_duration : float
        Duration of each epoch in seconds."""
    # Time
    L = len(probability)
    t = np.linspace(0, L * epochs_duration, L)

    # Estimate time of LoC
    # Least-squares : reduce the influence of outliers
    # t < 110 * 60 prevents divergence, LoC is assumed to occur before 2 hours
    res_robust_LOC = least_squares(
        fun_LOC,
        np.array([0.2, 15 * 60]),
        loss="soft_l1",
        f_scale=0.1,
        args=(t[t < 110 * 60],
              probability[t < 110 * 60],
              ),
        bounds=([0.01, 0.5], [5, 110 * 60]),
    )
    time_loc = res_robust_LOC.x[1]

    # Estimate time of RoC
    # Least-squares : reduce the influence of outliers
    # t > time_loc + min_dur_intervention * 60 prevents divergence
    min_dur_intervention = 30  # minutes
    res_robust_ROC = least_squares(
        fun_ROC,
        np.array([0.2, t[int(len(t) * 0.95)]]),
        loss="soft_l1",
        f_scale=0.1,
        args=(
            t[t > (time_loc + min_dur_intervention * 60)],
            probability[t > (time_loc + 30 * 60)],
        ),
        bounds=(
            [0.01, 0.5],
            [(time_loc + min_dur_intervention * 60), max(t)],
        ),
    )

    time_roc = res_robust_ROC.x[1]

    return time_loc, time_roc, t


def extract_loc_roc(raw):
    """Extract the LOC and ROC times from raw EEG data,
    using ONNX voting ensemble model.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.

    Returns
    -------
    time_loc : np.float
        Time of LoC in seconds.
    time_roc : np.float
        Time of RoC in seconds.
    t_proba : np.ndarray
        Time of each epoch in seconds.
    proba : np.ndarray
        Probability of each epoch.
    """
    # Pre-process data
    input_sample = compute_input_sample(raw).to_numpy()
    # Load onnx model
    session = onx_make_session('boost_loc_roc/model_weights/voting_model.onnx')
    # Predict probabilities
    proba = onx_predict_proba(session, input_sample)
    proba = smooth_probability(proba)
    # Infer times of LoC and RoC
    time_loc, time_roc, t_proba = proba_to_tloc_troc(proba, epochs_duration=30)
    return time_loc, time_roc, t_proba, proba


def extract_loc_roc_sklearn(
    raw: mne.io.Raw,
) -> tuple[np.float64, np.float64, np.ndarray, np.ndarray]:
    """
    Extract the LOC and ROC from raw EEG data,
    using sklearn pretrained model.

    To load the pretrained sklearn model,
    you need to have the same sklearn version as the one used to train the
    model, i.e 1.0.2. To avoid this constraint, use `extract_loc_roc` instead.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.

    Returns
    -------
    time_loc : np.float64
        Time of LoC in seconds.
    time_roc : np.float64
        Time of RoC in seconds.
    t : np.ndarray
        Time of each epoch in seconds.
    probability : np.ndarray
        Probability of each epoch.
    """
    voting_ensemble_model = load_voting_skmodel()

    input_samples = compute_input_sample(raw).to_numpy()

    probability = voting_ensemble_model.predict_proba(input_samples)
    probability = smooth_probability(probability)
    time_loc, time_roc, t = proba_to_tloc_troc(probability, epochs_duration=30)

    return time_loc, time_roc, t, probability
