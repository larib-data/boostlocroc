import mne
import numpy as np


def check_and_rescale_units(filename, threshold=5, new_filename=None):
    """
    Check the units of the EEG signal and rescale if necessary.
    
    Parameters:
    filename (str): Path to the input .fif file.
    threshold (float): The threshold to determine if the units are in microvolts (a bit arbitrary, but distinguishing between uV and V should be enough).
    new_filename (str): Path to save the rescaled .fif file. If None, a default name is used.
    
    Returns:
    str: Path to the rescaled .fif file, if rescaling was performed.
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
            new_filename = filename.replace('_eeg.fif', '_rescaled_eeg.fif')
        
        # Save the rescaled data to a new file
        raw.save(new_filename, overwrite=True)
        print(f"Rescaled data saved to {new_filename}")
        
        return new_filename
    else:
        print("The units are likely already in volts. No rescaling needed.")
        return None

    
def detrend_and_reset_time(df,  var='BIS/EEG1_WAV', new_name='EEG', trend_wind=300):

    # time reset 
    start = df['Time'].iloc[0] 
    df['Time'] = df['Time'] - start

    # soft detrend 
    dt = np.nanmedian(np.diff(df['Time'] ))
    df[var] = df[var].interpolate(method='linear')
    trend = df[var].rolling(window=int(trend_wind/dt), min_periods=1, center=True).mean()
    df[new_name] = df[var] - trend

    return df