import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from boostlocroc.eeg_features import compute_input_sample, smooth_probability
from boostlocroc.model import create_voting_ensemble_model, get_models


def fun_LOC(x, t, y):
    """Minimization of the logistic regression."""
    return (1 - x[2]) + x[2] * objective_LOC(t, x[0], x[1]) - y


def fun_ROC(x, t, y):
    """Minimization of the inverse of the logistic regression."""
    return (1 - x[2]) + x[2] * objective_ROC(t, x[0], x[1]) - y


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


def keep_last_segment_of_ones(series):
    # Initialize a flag to indicate if we are in the last segment of 1's
    in_last_segment = False

    # Traverse the series in reverse to find the last segment of 1's
    for idx in range(len(series) - 1, -1, -1):
        if series.iloc[idx] == 1:
            in_last_segment = True
        elif in_last_segment:
            break

    # Set all values before the last segment of 1's to zero
    if in_last_segment:
        series.iloc[: idx + 1] = 0

    return series


def extract_loc_roc(
    raw: mne.io.Raw,
    n_splits: int = 3,
    subsampling: bool = False,  # subsampling option
    weighted: bool = True,  # Weighted option
    binary: bool = False,  # True = 2 labels / False = 3 labels
    seed: int = 42,
) -> tuple[np.float64, np.float64, np.ndarray, np.ndarray]:
    """
    Extract the LOC and ROC from raw EEG data. (MAIN FUNCTION)

    Return LoC and RoC times in second (from the beginning of the operation)
    """

    option = define_option(subsampling, weighted, binary)

    cross_val_pathname = f"cross_val_weights_{seed}_{option}"
    models = get_models(cross_val_pathname, n_splits)

    voting_ensemble_model = create_voting_ensemble_model(models)

    epochs_duration = 30
    shift = 30
    n_fft = 512
    n_overlap = 128
    num_features = 50

    input_samples = compute_input_sample(
        raw, epochs_duration, shift, n_fft, n_overlap, num_features
    )

    probability = voting_ensemble_model.predict_proba(input_samples)
    probability = smooth_probability(probability)

    # !!! New component here
    eeg_signal = raw.get_data()[1, :] * 10**6
    mask = pd.Series(1 * (np.abs(eeg_signal) < 5))
    mask = mask.rolling(int(10 * 63), min_periods=1, center=True).min()
    mask = mask.rolling(int(15 * 63), min_periods=1, center=True).max()
    mask.iloc[-1] = 1
    mask = keep_last_segment_of_ones(mask)

    freq_raw = raw.info["sfreq"]
    n_samples = raw.n_times
    time_tmp = np.arange(n_samples) / freq_raw
    interp_mask = interp1d(
        time_tmp,
        mask,
        kind="linear",
        bounds_error=False,
        fill_value=(mask.iloc[0], mask.iloc[-1]),
    )

    # Time
    L = len(probability)
    t = np.linspace(0, L * epochs_duration, L)

    # probability correction for detached electrode at the end
    mask_probability = np.array(interp_mask(t))

    probability[mask_probability > 0.5] /= 2

    # Least-squares : reduce the influence of outliers
    t_initial = np.min([15 * 60, np.max(t) / 3])
    res_robust_LOC = least_squares(
        fun_LOC,
        np.array([0.4, t_initial, 1]),
        loss="soft_l1",
        f_scale=0.1,
        args=(
            t[t < 110 * 60],
            probability[t < 110 * 60],
        ),
        bounds=([0.02, 0.5, 0.5], [10, 110 * 60, 1]),
    )
    time_loc = res_robust_LOC.x[1]

    # Time
    L = len(probability)
    t = np.linspace(0, L * epochs_duration, L)
    min_dur_intervention = 30

    # Least-sqaures : reduce the influence of outliers
    res_robust_ROC = least_squares(
        fun_ROC,
        np.array([0.25, t[int(len(t) * 0.95)], 1]),
        loss="soft_l1",
        f_scale=0.1,
        args=(
            t[t > (time_loc + min_dur_intervention * 60)],
            probability[t > (time_loc + 30 * 60)],
        ),
        bounds=(
            [0.02, 0.5, 0.5],
            [10, max(t), 1],
        ),
    )

    time_roc = res_robust_ROC.x[1]

    LoC_params = res_robust_LOC.x
    RoC_params = res_robust_ROC.x

    return time_loc, time_roc, t, probability, LoC_params, RoC_params


def base_plot_spectrogram(
    time_loc,
    time_roc,
    signal,
    Fs,
    time,
    t_proba=None,
    proba=None,
    params_LoC=None,
    params_RoC=None,
    patient_id=None,
    debug=False,
):
    """Base function to plot spectrogram with optional probability and debug
    plots."""
    fig = plt.figure(figsize=(15, 6))

    if t_proba is not None and proba is not None:
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])

        # Create each subplot on the grid
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax2 = fig.add_subplot(gs[2], sharex=ax0)

        ax = [ax0, ax1, ax2]
        ax[2].scatter(t_proba, proba)
        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("Proba")

        if debug and params_LoC is not None and params_RoC is not None:
            ax[2].plot(
                t_proba,
                (1 - params_RoC[2])
                + params_RoC[2] * objective_LOC(t_proba, params_LoC[0], params_LoC[1]),
                color="r",
            )
            ax[2].plot(
                t_proba,
                (1 - params_RoC[2])
                + params_RoC[2] * objective_ROC(t_proba, params_RoC[0], params_RoC[1]),
                color="b",
                linestyle="--",
            )

    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])

        # Create each subplot on the grid
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)

        ax = [ax0, ax1]

    ax[0].axvline(x=time_loc, color="r", linestyle="--", linewidth=3)
    ax[0].axvline(x=time_roc, color="r", linestyle="--", linewidth=3)

    nperseg = np.floor(1.5 * Fs).squeeze()
    noverlap = np.floor(nperseg / 3).squeeze()

    pxx, freqs, bins, im = ax[0].specgram(
        signal,
        Fs=Fs,
        NFFT=int(nperseg),
        mode="psd",
        cmap="jet",
        noverlap=int(noverlap)
    )

    im.set_clim(-20, 25)

    ax[1].plot(time, signal)
    ax[1].axvline(x=time_loc, color="r", linestyle="--", linewidth=3)
    ax[1].axvline(x=time_roc, color="r", linestyle="--", linewidth=3)
    ax[1].set_ylim([-150, 150])

    # Hide x-axis labels for the shared subplots
    plt.setp(ax[0].get_xticklabels(), visible=False)
    if t_proba is not None and proba is not None:
        plt.setp(ax[1].get_xticklabels(), visible=False)

    plt.title(patient_id)
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()


def plot_spectrogram(time_loc, time_roc, signal, Fs, time, patient_id=None):
    """Plot the spectrogram and probability as subplots."""
    base_plot_spectrogram(
        time_loc,
        time_roc,
        signal,
        Fs,
        time,
        t_proba=None,
        proba=None,
        patient_id=patient_id,
    )


def plot_spectrogram_proba(
    time_loc, time_roc, signal, Fs, time, t_proba, proba, patient_id=None
):
    """Plot the spectrogram with probabilities."""
    base_plot_spectrogram(
        time_loc,
        time_roc,
        signal,
        Fs,
        time,
        t_proba,
        proba,
        params_LoC=None,
        params_RoC=None,
        patient_id=patient_id,
    )


def plot_spectrogram_debug(
    time_loc,
    time_roc,
    signal,
    Fs,
    time,
    t_proba,
    proba,
    params_LoC,
    params_RoC,
    patient_id=None,
):
    """DEBUG TOOL: Similar to plot_spectrogram with additional sigmoids
    visible for inspection."""
    base_plot_spectrogram(
        time_loc,
        time_roc,
        signal,
        Fs,
        time,
        t_proba,
        proba,
        params_LoC=params_LoC,
        params_RoC=params_RoC,
        patient_id=patient_id,
        debug=True,
    )


def Truncate_fif(raw, electrode=1):
    """Remove data from a Raw object where the signal is between -0.1 and 0.1
    uV."""

    data, times = raw[:]
    electrode = data[electrode, :] * 10**6

    mask1 = (electrode > -1) & (electrode < 1)
    mask1 = 1 - mask1
    mask2 = np.ones_like(mask1)
    first_one = np.argmax(mask1)
    last_one = len(mask1) - 1 - np.argmax(mask1[::-1])

    mask2[:first_one] = 0
    mask2[last_one + 1:] = 0
    mask2 = mask2.astype(bool)

    # data[:, ~mask2] = 0
    data = data[:, mask2]
    ch_types = raw.get_channel_types()

    info = mne.create_info(
        ch_names=raw.info["ch_names"],
        sfreq=raw.info["sfreq"],
        ch_types=ch_types
    )

    return mne.io.RawArray(data, info)
