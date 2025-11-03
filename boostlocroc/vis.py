"""Code for figure plots and visualizations.

Functions
    base_plot_spectrogram
    plot_spectrogram
    plot_spectrogram_proba
    plot_spectrogram_debug

"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from boostlocroc.main import objective_LOC, objective_ROC


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

