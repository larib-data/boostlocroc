"""
The eeg_features module.

Functions
---------
    create_validation_set,
    raw_segmentation,
    smooth_psd,
    compute_input_sample,
    fit_gbc,
    predict_gbc,
    smooth_probability,
    compute_probability,
    find_loc,
    find_roc,
    display_loc_roc,


"""
import random

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def create_validation_set(dataset):
    """
    Create validation set using the train_test_split function from sklearn.model_selection. 
    Statistical consistency is preserved by using stratification on labels.

    Parameters
    ----------
     dataset: pd.dataframe
         The first column must be the patient ID and the last one the labels.

    Return
    ------
     X_train, X_test, y_train, y_test, X_validation, y_validation : DataFrames
    """
    print("Create dataset")
    # Construction of the validation set
    patients_id = np.unique(dataset.iloc[:, 0])
    n_validation = int(0.2 * len(patients_id))
    validation_id = random.sample(list(patients_id), n_validation)
    validation_set = dataset.loc[dataset.iloc[:, 0].isin(validation_id)]

    # Set to train and test
    new_set = dataset.loc[-dataset.iloc[:, 0].isin(validation_id)]
    assert len(new_set) + len(validation_set) == len(dataset)

    # Validation set
    validation_set = validation_set.dropna()
    X_validation = validation_set.drop(validation_set.columns[[0, -1]], axis=1)
    y_validation = validation_set.iloc[:, -1]

    new_set = new_set.dropna()
    X_predictors = new_set.drop(new_set.columns[[0, -1]], axis=1)
    y_labels = new_set.iloc[:, -1]

    # make test and train sets using stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_predictors,
        y_labels,
        test_size=0.15,
        stratify=y_labels,
        random_state=42,
    )


def raw_segmentation(
    raw: mne.io.Raw,  
    epochs_duration: int, 
    shift: int,
) -> mne.Epochs:
    """
    Create epochs in order to segment the raw signal.

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
    # raw = raw.np.copy()

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
        flat={"eeg": 1e-7}, # remove epochs with signal amplitude between [-0.1;0.1] uV
        decim=1,
        verbose=False,
    )
    return epochs


def smooth_psd(
    epochs: mne.Epochs, 
    n_fft: int, 
    n_overlap: int, 
    num_features:int
) -> np.ndarray:
    """
    Compute the power spectral density (PSD) using Welch method for each epochs.
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
        epochs.get_data(),
        sfreq=63,
        fmin=0,
        fmax=30,
        n_fft=n_fft,
        n_overlap=n_overlap,
        average="median",
    )

    n, m = psd[:, 1, :].shape
    f_int = np.linspace(0.5, 25, num=num_features, endpoint=True)

    smooth_psd = np.zeros((num_features, n))

    for i in range(0, n):
        from numpy import ma

        psds = savgol_filter(10 * ma.log10(psd[i, 1, :]).filled(0) + 120, 5, 0)
        f_interpolation = interp1d(freqs, psds)
        psd_interpolate = f_interpolation(f_int)
        smooth_psd[:, i] = smooth_psd[:, i] + psd_interpolate

    return smooth_psd.T


def __objective_LOC(x, a, x0):
    """Logistic regression."""
    b = -a * (x - x0)
    b = np.where(b > 500, 500, b)
    return (1 + np.exp(b)) ** (-1)


def __fun_LOC(x, t, y):
    """Minimization of the logistic regression."""
    return __objective_LOC(t, x[0], x[1]) - y


def __objective_ROC(x, a, x0):
    """1 - Logistic regression."""
    b = -a * (x - x0)
    b = np.where(b > 500, 500, b)
    return 1 - (1 + np.exp(b)) ** (-1)


def __fun_ROC(x, t, y):
    """Minimization of the inverse of the logistic regression."""
    return __objective_ROC(t, x[0], x[1]) - y


def __tanh_soft(x, a, x0):
    """Hyperbolique tangent function."""
    return (np.tanh(a * (x - x0)) + 1) / 2


def compute_input_sample(
    raw: mne.io.Raw, 
    epochs_duration: int, 
    shift: int, 
    n_fft: int,
    n_overlap: int,
    num_features: int
) -> pd.DataFrame:
    """
    Transform the raw signal into sequential psd.

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

    # print(input_samples)
    return input_samples


def fit_gbc(X_train, y_train, save_weight):
    """
    Compute a GradientBoostingClassifieur. The choice of the parameters comes from the sklearn.model_selection.HalvingRandomSearchCV function.

    Parameters
    ----------
     X_train, y_train: ndarray of shape (n_samples, n_features)
         The input samples.

    Return
    ------
     class probabilities for input_samples : ndarray of shape(n_epochs, n_classes)
    """
    print("Fit GradientBoostingClassifier")
    gbc = GradientBoostingClassifier(
        n_estimators=138,
        learning_rate=0.24210526315789474,
        min_samples_leaf=6,
        max_depth=2,
        min_samples_split=2,
        random_state=0,
    )
    gbc.fit(X_train.values, y_train.values)
    dump(gbc, save_weight, compress=1)


def predict_gbc(input_samples, save_weight):
    """Use the gradient boosting classifier to predict on our test data."""
    gbc = load(save_weight)
    prediction = gbc.predict_proba(input_samples)
    return prediction


def load_labels():
    """Load labels."""
    Freq_alpha2 = pd.read_excel(
        "~/loc_roc_labels/Labeled_features_induction_Batch_2.xlsx"
    )
    Freq_alpha = pd.read_excel("~/loc_roc_labels/Test_Tensorflow_train2.xlsx")
    Freq_alpha = Freq_alpha.rename(columns={"ID": "patient"})
    Freq_alpha2 = Freq_alpha2.rename(columns={"Unnamed: 0": "patient"})
    Freq_alpha2 = pd.concat([Freq_alpha2, Freq_alpha], axis=0)
    return Freq_alpha2


def cross_validation(weights_path="save_weight.pkl", fit=True):
    """Source : https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import cross_validate
    import os

    freq_alpha = load_labels()  # TODO and review create_validation_set
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_validation,
        y_validation,
    ) = create_validation_set(freq_alpha)
    # print(X_test)
    if fit or not os.path.exists("save_weight.pkl"):
        fit_gbc(X_train, y_train, weights_path)

    model = load(weights_path)

    # define the evaluation method
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # evaluate the model on the dataset
    print("yyyyyyyyyyyyyyyy", y_test)
    n_scores = cross_validate(
        model,
        X_test,
        y_test,
        scoring=["balanced_accuracy", "roc_auc"],
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    print(n_scores)


def smooth_probability(probability):
    """
    Apply a Savitzky-Golay filter to the predict class probabilities compute for each epochs.

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


def compute_probability(
    raw,
    save_weight,
    epochs_duration=30,
    shift=30,
    n_fft=512,
    n_overlap=128,
    num_features=50,
):
    """
    Compute the probability.

    Parameters
    ----------
     raw : mne raw object
         An instance of raw.
     save_weight : pickle
         Weights of the classifier.
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
     smooth_probability : ndarray, shape(n_epochs, smooth_probability)
    """
    input_samples = compute_input_sample(
        raw, epochs_duration, shift, n_fft, n_overlap, num_features
    )

    loc_probability = predict_gbc(input_samples, save_weight)
    probability = smooth_probability(loc_probability)

    return probability


def find_loc(
    raw,
    save_weight,
    epochs_duration=30,
    shift=30,
    n_fft=512,
    n_overlap=128,
    num_features=50,
    display_proba=False,
):
    """
    Find the time of lost of consciousness.

    Parameters
    ----------
     raw : mne raw object
         An instance of raw.
     save_weight : pickle
         Weights of the classifier.
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
     loc : float
         time of lost of consciousness
    """
    probability = compute_probability(
        raw,
        save_weight,
        epochs_duration,
        shift,
        n_fft,
        n_overlap,
        num_features,
    )

    # Time
    L = len(probability)
    t = np.linspace(0, L * epochs_duration, L)

    # Least-sqaures : reduce the influence of outliers
    res_robust_LOC = least_squares(
        __fun_LOC,
        np.array([0.2, 15 * 60]),
        loss="soft_l1",
        f_scale=0.1,
        args=(
            t[t < 110 * 60],
            probability[t < 110 * 60],
        ),
        bounds=([0.005, 0.5], [5, 110 * 60]),
    )

    time_loc = res_robust_LOC.x[1]

    if display_proba:
        Fit_Patient_LOC = __objective_LOC(t, res_robust_LOC.x[0], res_robust_LOC.x[1])
        plt.plot(Fit_Patient_LOC)

    return time_loc


def find_roc(
    raw,
    save_weight,
    time_loc,
    epochs_duration=30,
    shift=30,
    n_fft=512,
    n_overlap=128,
    num_features=50,
    display_proba=False,
):
    """
    Find the time of recovery of consciousness(roc).

    Parameters
    ----------
     raw : mne raw object
         An instance of raw.
     save_weight : pickle
         Weights of the classifier.
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
     roc : float
        time of recovery of consciousness
    """
    probability = compute_probability(
        raw,
        save_weight,
        epochs_duration,
        shift,
        n_fft,
        n_overlap,
        num_features,
    )

    # Time
    L = len(probability)
    t = np.linspace(0, L * epochs_duration, L)
    min_dur_intervention = 30

    # Least-sqaures : reduce the influence of outliers
    res_robust_ROC = least_squares(
        __fun_ROC,
        np.array([0.2, t[int(len(t) * 0.95)]]),
        loss="soft_l1",
        f_scale=0.1,
        args=(
            t[t > (time_loc + min_dur_intervention * 60)],
            probability[t > (time_loc + 30 * 60)],
        ),
        bounds=(
            [0.005, 0.5],
            [(time_loc + min_dur_intervention * 60), max(t)],
        ),
    )

    time_roc = res_robust_ROC.x[1]

    if display_proba:
        Fit_Patient_ROC = __objective_ROC(t, res_robust_ROC.x[0], res_robust_ROC.x[1])
        plt.plot(Fit_Patient_ROC)

    return time_roc


def compute_accuracy(y_true, y_pred, precision):
    """Precision in seconds."""
    if y_pred < y_true - precision * 2 or y_pred > y_true + precision * 2:
        return 0.0

    x = [
        y_true - precision * 2,
        y_true - precision,
        y_true - precision / 2,
        y_true,
        y_true + precision,
        y_true + precision / 2,
        y_true + precision * 2,
    ]
    y = [0, 0.5, 0.9, 1, 0.9, 0.5, 0]

    return interp1d(x, y)(y_pred)


def display_loc_roc(
    data_patient,
    sample_freq,
    time_loc,
    time_roc,
    loc_probability,
    ch_name="EEG R1(Fp2)",
):
    """
    Display a spectrogram with the loc and roc calculated. Display the probability of beeing awake or not.

    Parameters
    ----------
     data_patient : pandas Dataframe
         The dataframe must contains column name starting with "EEG".
     ch_name : string
         Channel name, default is "EEG R1(Fp2)"
     sample_freq : float
         Sample frequency
     time_loc : float
     time_roc : float
     loc_probability : ndarray

    Return
    ------
     loc : float

    TODO
    ------
     add an option for the 10**6.
    """
    eeg = data_patient[[col for col in data_patient if col.startswith("EEG")]]
    df_raws = eeg[ch_name]
    df_raws = df_raws[~df_raws.isnull()]
    nperseg = np.floor(1.5 * sample_freq)
    noverlap = np.floor(nperseg / 3)

    ######
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(20)
    fig.suptitle("EEG feature extraction : LOC and ROC")

    ax1 = plt.subplot2grid(shape=(4, 3), loc=(0, 0), colspan=3)
    ax2 = plt.subplot2grid(shape=(4, 3), loc=(1, 0), colspan=3)

    # Ax1 spectrogram
    Axspectrum, freqs, bins, im = ax1.specgram(
        df_raws,
        Fs=sample_freq,
        NFFT=int(nperseg),
        mode="psd",
        cmap="jet",
        noverlap=noverlap,
    )

    ax1.vlines(time_loc, 0, 30, "w", "--", linewidth=5, label="LOC")
    ax1.vlines(time_roc, 0, 30, "w", "--", linewidth=5, label="ROC")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [sec]")
    im.set_clim(-25, 25)
    ax1.set_title("Spectrogram")
    ax1.set_xlim(0)
    ax1.set_ylim(0, 30)

    # Ax2 probability
    ax2.plot(loc_probability)
    ax2.set_ylabel("Sleep probability")
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(0, len(loc_probability))
    ax2.set_title("Probability")

    fig.tight_layout()

    plt.show()
