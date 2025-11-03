"""Preprocess EEG data.

- truncate_fif: Remove artefacted data from a mne.Raw object.
- detrend_and_reset_time: Reset time of a pandas.DataFrame and detrend one
column.
"""
import mne
import numpy as np


def truncate_fif(raw, electrode=1):
    """Remove data from a Raw object where the signal is almost 0.

    The signal is considered close to zero if it is between
    -0.1 and 0.1 uV. The EEG signal in the raw object is expected to be in
    volts.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    electrode : int
        Index of the electrode to consider.

    Returns
    -------
    mne.io.Raw
        Raw EEG data with the signal between -1 and 1 uV removed.
    """
    data, times = raw[:]
    electrode = data[electrode, :]*10**6  # Convert to uV

    mask1 = (electrode > -1) & (electrode < 1)
    mask1 = 1 - mask1
    mask2 = np.ones_like(mask1)
    first_one = np.argmax(mask1)
    last_one = len(mask1) - 1 - np.argmax(mask1[::-1])

    mask2[:first_one] = 0
    mask2[last_one + 1:] = 0
    mask2 = mask2.astype(bool)

    data = data[:, mask2]
    ch_types = [
        mne.channel_type(raw.info, idx) for idx in range(raw.info['nchan'])
    ]

    info = mne.create_info(
        ch_names=raw.info['ch_names'],
        sfreq=raw.info['sfreq'],
        ch_types=ch_types
    )

    return mne.io.RawArray(data, info)


def detrend_and_reset_time(df, var="BIS/EEG1_WAV", new_name="EEG",
                           trend_wind=300):
    """
    Detrends channel `var` of `df` DataFrame and and resets its time so it
    starts at 0 seconds.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame containing the data. df is modified *in place*.
        Must contain channels "Time" and `var`.
    var: str
        The name of the channel to detrend. Default is "BIS/EEG1_WAV".
    new_name: str
        The new name for the detrended channel. Default is "EEG".
    trend_wind: int
        The window size for the soft detrending. Default is 300.

    Returns
    -------
    df: pandas.DataFrame
        The modified DataFrame where time is reset,
        channel `var` is replaced with its interpolation, and
        the detrended channel is added with name `new_name`.
    """
    # Time reset
    start = df["Time"].iloc[0]
    df["Time"] = df["Time"] - start

    # Soft detrend
    dt = np.nanmedian(np.diff(df["Time"]))
    df[var] = df[var].interpolate(method="linear")
    window = int(trend_wind / dt)
    trend = df[var].rolling(window=window, min_periods=1, center=True)
    trend = trend.mean()
    df[new_name] = df[var] - trend

    return df


def check_and_rescale_units(filename, threshold=5, new_filename=None):
    """
    Check the units of the EEG signal and rescale if necessary.

    Parameters
    ----------
    filename: str
        Path to the input .fif file.
    threshold: float, default=5
        The threshold to determine if the units are in microvolts.
        If the median of the absolute values of the signal exceeds this
        threshold, the units are likely in microvolts and the signal is
        rescaled to volts.
    new_filename: str, default=None
        Path to save the rescaled .fif file. If None, a default name is used.

    Returns
    -------
    res: str or None.
        Path to the rescaled .fif file, if rescaling was performed.
        None, if no rescaling was performed.
    """
    # Read the raw data
    raw = mne.io.read_raw_fif(filename, preload=True)

    # Compute the median of the absolute values of the signal
    median_value = np.median(np.abs(raw._data))
    # print(f"Median of absolute signal values: {median_value}")

    # Check if the median value exceeds the threshold
    if median_value > threshold:
        print("The units are likely in microvolts. Rescaling to volts.")

        # Rescale the signal
        raw._data *= 10**-6

        # Define the new filename if not provided
        if new_filename is None:
            new_filename = filename.replace("_eeg.fif", "_rescaled_eeg.fif")

        # Save the rescaled data to a new file
        raw.save(new_filename, overwrite=True)
        print(f"Rescaled data saved to {new_filename}")

        return new_filename
    else:
        print("The units are likely already in volts. No rescaling needed.")
        return None
