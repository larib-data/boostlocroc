"""Preprocess EEG data.

- truncate_fif: Remove artefacted data from a mne.Raw object.
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
