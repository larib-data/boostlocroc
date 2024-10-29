"""TODO : add description."""

sfreq = 63


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
