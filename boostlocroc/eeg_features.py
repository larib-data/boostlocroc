"""Functions for computing EEG features.

Functions
---------
    smooth_psd,
    compute_input_sample,
    smooth_probability,
"""
import mne
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def raw_segmentation(
    raw: mne.io.Raw,
    epochs_duration: int,
    shift: int,
) -> mne.Epochs:
    """Create epochs in order to segment the raw signal.

    Parameters
    ----------
     raw : mne raw object.
         An instance of raw.
     epochs_duration : int
         Epoch duration in seconds.
     shift : int
         The duration to separate events by (in seconds).

    Return
    ------
     Epochs : mne object

    """
    # Creating Epochs/events
    events = mne.make_fixed_length_events(
        raw,
        id=3000,
        start=0,
        duration=shift,
        stop=raw.times[-1] - epochs_duration,
    )
    epochs = mne.Epochs(
        raw,
        events,
        event_id=3000,
        tmin=0,
        tmax=epochs_duration,
        proj=True,
        baseline=None,
        reject=None,
        preload=True,
        flat={"eeg": 1e-7},  # remove epochs with signal between [-0.1;0.1] uV
        decim=1,
        verbose=False,
    )
    return epochs


def smooth_psd(
    epochs: mne.Epochs,
    n_fft: int,
    n_overlap: int,
    num_features: int,
) -> np.ndarray:
    """Compute the smooth PSD using Welch method for each epoch.

    Apply a Savitzky-Golay filter to the PSD computed for each epochs.

    Parameters
    ----------
     Epochs : mne object
     n_fft : int
         The length of FFT used, must be <= number of time points in the data.
     n_overlap : int
         The number of points of overlap between segments.
     num_features : int
         The number of features selected.

    Return
    ------
     Smooth_psd : ndarray, shape(n_epochs, smooth_psd)

    """
    psd, freqs = mne.time_frequency.psd_array_welch(
        epochs.get_data(copy=False),
        sfreq=epochs.info["sfreq"],
        fmin=0,
        fmax=30,
        n_fft=n_fft,
        n_overlap=n_overlap,
        average="median",
    )

    n, m = psd[:, 1, :].shape
    f_int = np.linspace(0.5, 25, num=num_features, endpoint=True)

    smooth_psd = np.zeros((num_features, n))

    for i in range(n):
        from numpy import ma

        psds = savgol_filter(10 * ma.log10(psd[i, 1, :]).filled(0) + 120, 5, 0)
        f_interpolation = interp1d(freqs, psds)
        psd_interpolate = f_interpolation(f_int)
        smooth_psd[:, i] = smooth_psd[:, i] + psd_interpolate

    return smooth_psd.T


def __tanh_soft(x, a, x0):
    """Hyperbolique tangent function."""
    return (np.tanh(a * (x - x0)) + 1) / 2


def compute_input_sample(
    raw: mne.io.Raw,
    epochs_duration: int = 30,
    shift: int = 30,
    n_fft: int = 512,
    n_overlap: int = 128,
    num_features: int = 50,
) -> pd.DataFrame:
    """Transform the raw signal into sequential psd.

    Parameters
    ----------
     raw : mne raw object.
         An instance of raw.
     epochs_duration : int
         Epoch duration in seconds.
     shift : int
         The duration to separate events by (in seconds).
     n_fft : int
         The length of FFT used, must be <= number of time points in the data.
     n_overlap : int
         The number of points of overlap between segments.
     num_features : int
         The number of features selected.

    Return
    ------
     smooth_psd : ndarray, shape(n_epochs, smooth_psd)

    """
    epochs = raw_segmentation(raw, epochs_duration, shift)
    input_samples = smooth_psd(epochs, n_fft, n_overlap, num_features)

    input_samples = pd.DataFrame(input_samples)

    return input_samples


def smooth_probability(probability):
    """Apply a Savitzky-Golay filter to the predict class probabilities compute
    for each epochs.

    Parameters
    ----------
     predict class probabilities : ndarray

    Return
    ------
     smooth_probability : ndarray, shape(n_epochs, smooth_probability)

    """
    smooth_probability = savgol_filter(probability[:, 1], 3, 1)
    smooth_probability = __tanh_soft(smooth_probability, 2, 0.5)

    return smooth_probability
