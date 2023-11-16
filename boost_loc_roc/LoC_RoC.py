import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
from scipy.optimize import least_squares
from boost_loc_roc.model import get_models, create_voting_ensemble_model
from boost_loc_roc.eeg_features import smooth_probability, compute_input_sample


def fun_LOC(x, t, y):
    """Minimization of the logistic regression."""
    return objective_LOC(t, x[0], x[1]) - y


def fun_ROC(x, t, y):
    """Minimization of the inverse of the logistic regression."""
    return objective_ROC(t, x[0], x[1]) - y


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
    res ="--"
    if subsampling:
        res += "s"
    if weighted:
        res += "w"
    if binary:
        res += "b"
    res+="--"
    return res


def extract_loc_roc(raw):
    """Extract the LOC and ROC from raw EEG data. (MAIN FUNCTION)"""
    n_splits = 3
    subsampling = False # subsampling option
    weighted = True # Weighted option 
    binary = False # True = 2 labels / False = 3 labels
    seed = 42
    metrics_each_models = False
    precision = 40

    option = define_option(subsampling, weighted, binary)

    cross_val_pathname = f"cross_val_weights_{seed}_{option}"
    weights_dir = "boost_loc_roc/model_weights/"
    models = get_models(weights_dir,cross_val_pathname, n_splits)
    
    voting_ensemble_model = create_voting_ensemble_model(models, weights_dir)
    # print(voting_ensemble_model)
    
    epochs_duration=30
    shift=30
    n_fft=512
    n_overlap=128
    num_features=50

    
    input_samples = compute_input_sample(raw, 
                                     epochs_duration, 
                                     shift, 
                                     n_fft, 
                                     n_overlap,
                                     num_features)

    probability = voting_ensemble_model.predict_proba(input_samples)
    probability = smooth_probability(probability)

     # Time
    L = len(probability)
    t = np.linspace(0, L * epochs_duration, L)

    # Least-squares : reduce the influence of outliers
    res_robust_LOC = least_squares(
        fun_LOC,
        np.array([0.2, 15 * 60]),
        loss="soft_l1",
        f_scale=0.1,
        args=(t[t < 110 * 60],
              probability[t < 110 * 60],
             ),
        bounds=([0.01, 0.5], [5, 110 * 60]),
    )
    time_loc = res_robust_LOC.x[1]

    # Time
    L = len(probability)
    t = np.linspace(0, L * epochs_duration, L)
    min_dur_intervention = 30

    # Least-sqaures : reduce the influence of outliers
    res_robust_ROC = least_squares(
        fun_ROC,
        np.array([0.2, t[int(len(t) * 0.95)]]),
        loss="soft_l1",
        f_scale=0.1,
        args=(
            t[t > (time_loc + min_dur_intervention * 60)],
            probability[t > (time_loc + 30 * 60)],
        ),
        bounds=(
            [0.01, 0.5],
            [(time_loc + min_dur_intervention * 60), max(t)],
        ),
    )

    time_roc = res_robust_ROC.x[1]

    return time_loc, time_roc, t, probability


def plot_spectrogram(time_loc, time_roc, signal, Fs, time, t_proba, proba): 
    """Plot the spectrogram and probability as subplots."""
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])

    # Create each subplot on the grid
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    
    ax = [ax0, ax1, ax2]
    ax[0].axvline(x=time_loc, color='r', linestyle='--', linewidth=3)
    ax[0].axvline(x=time_roc, color='r', linestyle='--', linewidth=3)
    
    nperseg = np.floor(1.5*Fs).squeeze() 
    noverlap = np.floor(nperseg/3).squeeze() 
    
    pxx, freqs, bins, im = ax[0].specgram(
        signal, 
        Fs = Fs, 
        NFFT = int(nperseg),
        # window= np.hamming,
        mode= 'psd',
        cmap= 'jet',
        noverlap = int(noverlap))
    
    im.set_clim(-20,25)

    ax[1].plot(time, signal)
    ax[1].axvline(x=time_loc, color='r', linestyle='--', linewidth=3)
    ax[1].axvline(x=time_roc, color='r', linestyle='--', linewidth=3)
    
    ax[2].scatter(t_proba, proba)
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Proba')
    plt.show()
    
    
def Truncate_fif(raw, electrode=1):
    """Remove data from a Raw object where the signal is between -0.1 and 0.1 uV."""

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
    ch_types = [mne.io.pick.channel_type(raw.info, idx) for idx in range(raw.info['nchan'])]
    
    info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=raw.info['sfreq'], ch_types=ch_types)

    return mne.io.RawArray(data, info)