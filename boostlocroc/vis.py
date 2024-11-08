"""Code for figure plots and visualizations.

Functions
    plot_spectrogram
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(time_loc, time_roc, signal, sfreq, time, t_proba, proba):
    """Plots the spectrogram, estimated LoC/RoC times and per epoch LoC/RoC
    probability.

    Parameters
    ----------
    time_loc : float
        Time of LoC in seconds.
    time_roc : float
        Time of RoC in seconds.
    signal : np.ndarray
        EEG signal to plot.
    sfreq: float
        Sampling frequency of the EEG signal.
    time : np.ndarray
        Time of each EEG sample in seconds.
    t_proba: np.ndarray
        Time of each epoch in seconds.
    proba: np.ndarray
        Probability of each epoch.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])

    # Create each subplot on the grid
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    ax = [ax0, ax1, ax2]
    ax[0].axvline(x=time_loc, color="r", linestyle="--", linewidth=3)
    ax[0].axvline(x=time_roc, color="r", linestyle="--", linewidth=3)

    nperseg = np.floor(1.5 * sfreq).squeeze()
    noverlap = np.floor(nperseg / 3).squeeze()
    # Plot spectrogram
    pxx, freqs, bins, im = ax[0].specgram(
        signal,
        Fs=sfreq,
        NFFT=int(nperseg),
        mode="psd",
        cmap="jet",
        noverlap=int(noverlap),
    )

    im.set_clim(-20, 25)
    # Add lines at predicted times of LoC and RoC
    ax[1].plot(time, signal)
    ax[1].axvline(x=time_loc, color="r", linestyle="--", linewidth=3)
    ax[1].axvline(x=time_roc, color="r", linestyle="--", linewidth=3)
    # Plot classification probabilities
    ax[2].scatter(t_proba, proba)
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Proba")
    plt.show()
