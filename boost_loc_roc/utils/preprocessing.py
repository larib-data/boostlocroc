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
    mask1 = 1-mask1
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


def filter_operation(df, raw, y_true):
    """Filter the EEG."""
    parquet, (first, last) = __filter_df_between(df)
    loc, roc = y_true
    loc -= first / sfreq
    roc -= first / sfreq

    raw = raw.copy().crop(first / sfreq, last / sfreq, include_tmax=False)
    return (loc, roc), parquet, raw


def __filter_df_between(df, low=-0.02, high=0.02):
    """Filter the EEG when the device is already on."""
    condition = (df < low) | (df > high)
    condition = df[condition.all(axis=1)]

    valid_signal_index = (
        df.index.get_loc(condition.first_valid_index()),
        df.index.get_loc(condition.last_valid_index()),
    )

    return df.iloc[valid_signal_index[0] : valid_signal_index[1]], valid_signal_index
