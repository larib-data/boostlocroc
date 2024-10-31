"""Preprocess EEG data."""
import numpy as np
import mne

sfreq = 63


def truncate_fif(raw, electrode=1):
    """Remove data from a Raw object where the signal is between
    -0.1 and 0.1 uV."""

    data, times = raw[:]
    electrode = data[electrode, :]*10**6

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
    ch_types = [
        mne.channel_type(raw.info, idx) for idx in range(raw.info['nchan'])
    ]

    info = mne.create_info(ch_names=raw.info['ch_names'],
                           sfreq=raw.info['sfreq'], ch_types=ch_types)

    return mne.io.RawArray(data, info)
