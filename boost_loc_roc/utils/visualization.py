"""Visualization module."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import os.path as op
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams

sfreq =  63

params = {
    "figure.figsize": (20, 5),
    "figure.constrained_layout.use": True,
    "axes.titlepad": 12,
    "axes.titlesize": 20,
    "xtick.labelsize": 8,
    "ytick.labelsize": 12,
    "legend.fontsize": 15,
    "figure.titlesize": 25,
    "axes.labelsize": 15,
    "image.cmap": "jet",
    "legend.loc": "upper center",
}

rcParams.update(params)

def get_band_signal(
    raw,
    sample_freq,
    fmin,
    fmax,
    column_name,
    display=False,
    verbose=False,
    ax=None,
):
    """
    Get band signal between two frequencies.

    Parameters
    ----------
     raw : :obj:`mne.io.edf.edf.RawEDF`
         Raw object of mne

     sample_freq : int
         The sample frequency.

     fmin : int
         Minimum frequency to get.

     fmax : int
         Maximum frequency to get.

     column_name : str
         The name of the column to get from the raw object.

     display: bool, optional, default: False
         Display the signal between the two given frequencies.

     verbose: bool, optional, default: False
         Verbose argument of mne load_data and filter functions.

     shift: int, optional, default: 15
         Duration of an events (see mne.Events).

    Returns
    -------
     band : :obj:`pandas.core.frame.DataFrame`
         The signal between the two frequencies.
    """
    raw = raw.copy()
    band = (
        raw.load_data(verbose=verbose)
        .filter(l_freq=fmin, h_freq=fmax, verbose=verbose)
        .to_data_frame()
        .drop(["time"], axis=1)[column_name]
        .to_frame()
    )

    band.index = band.reset_index(drop=True).index / sample_freq

    if ax is not None:
        ax.plot(band)
    elif display:
        plt.figure()
        plt.plot(band)
        plt.title(f"Band signal between {fmin} and {fmax}")
        plt.show()

    return band


def add_ax_vlines(ax, y_true, y_pred, display_target):
    """Visualisation tool."""
    ax.vlines(
        [y_true[0], y_true[1]], 0, 30, "r", ":", linewidth=4, label="LOC & ROC target"
    )

    if display_target:
        ax.vlines(
            [y_pred[0], y_pred[1]],
            0,
            30,
            "b",
            ":",
            linewidth=4,
            label="LOC & ROC prediction",
        )


def add_vspans(ax, y_true, precision):
    """Visualisation tool."""
    loc, roc = y_true
    ax.axvspan(roc - precision, roc + precision, facecolor="r", alpha=0.2)
    ax.axvspan(loc - precision, loc + precision, facecolor="r", alpha=0.2)


def plot_eeg(df, raw, low_f, high_f, y_true, y_pred, ax=None, display_target=True):
    """Visualisation tool."""
    if ax is None:
        ax = plt.axes()

    get_band_signal(raw, sfreq, low_f, high_f, "Fp2", display=True, ax=ax)

    set_ax_setting(ax, df.index[0], df.index[-1])
    add_ax_vlines(ax, y_true, y_pred, display_target)

############################
def set_ax_setting(ax, min_x, max_x, ylim=(-30, 30), step=250, labelrotation=45):
    """Visualisation tool."""
    ax.set_xticks(np.arange(min_x, max_x, step))
    ax.tick_params(axis="x", labelrotation=labelrotation)
    ax.set_xlim(min_x, max_x)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        
def plot_proba (proba_raw, proba_smooth, bedstay):
    fig, ax = plt.subplots()
    ax.plot(proba_smooth, label = 'smooth probability')
    ax.plot(proba_raw, label = 'voting ensemble model probabilities')
    ax.legend(loc = 'center')
    ax.set_title(
        f"Voting ensemble probabilities of Loc & ROC for the patient {bedstay}"
    )
    plt.show()

def plot_spectro(
    df,
    y_true,
    y_pred,
    subject_name,
    sfreq = 63,
):
    """Visualisation tool to display the prediction of LOC and ROC on a spectrogram"""
    nperseg = np.floor(1.5 * sfreq)
    noverlap = np.floor(nperseg / 3)

    ax = plt.axes()

    # Spectrogram
    Axspectrum, freqs, bins, im = ax.specgram(
        df.iloc[:,2],
        Fs=sfreq,
        NFFT=int(nperseg),
        mode="psd",
        noverlap=noverlap,
    )
    im.set_clim(-25, 25)
    ax.set_title(
        f"EEG feature extraction : LOC and ROC\n\nSpectrogram of {subject_name}"
    )
    ax.vlines(
    [y_true, y_pred], 0, 30, "r", ":", linewidth=4, label="LOC & ROC prediction"
    )
    set_ax_setting(ax, df.index[0], df.index[-1] / sfreq, ylim=(0, 30))

def make_confusion_matrix(
    y_test,
    y_pred,
    normalized_type="pred",
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="YlGnBu",
    title=None,
    save_title=None,
    directory="images",
):
    """
    Plot an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
    title:         Title for the heatmap. Default is None.
    """
    # CODE TO GENERATE THE MATRIX
    cf = confusion_matrix(y_test, y_pred, normalize="pred")
    cf_matrix = confusion_matrix(y_test, y_pred)

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf_matrix.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf_matrix) / float(np.sum(cf_matrix))

        # if it is a binary confusion matrix, show some more stats
        if len(cf_matrix) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf_matrix[1, 1] / sum(cf_matrix[:, 1])
            recall = cf_matrix[1, 1] / sum(cf_matrix[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.4f}\nPrecision={:0.4f}\nRecall={:0.4f}\nF1 Score={:0.4f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.4f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks is False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.savefig(op.join(directory, f"CM_{save_title}.svg"))

    return accuracy


def display_roc_proba(
    X_test, y_test, model, binary=True, save_title=None, directory="images"
):
    """Visualisation tool."""
    y_proba = model.predict_proba(X_test)

    if binary:
        n_classes = 2
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.figure(figsize=(8, 8))

        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            label=roc_auc,  # "AUC = {1:0.3f}" "".format(1, roc_auc),
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic Curve")
        plt.legend(loc="lower right")

        plt.savefig(op.join(directory, f"ROC_{save_title}.svg"))

    else:
        y_test = label_binarize(y_test, classes=[0, 1, 2])

        n_classes = y_test.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.figure(figsize=(8, 8))
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.3f})"
            "".format(roc_auc["micro"]),
        )
        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                label="ROC curve of class {0} (area = {1:0.3f})"
                "".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic Curve")
        plt.legend(loc="lower right")

        plt.savefig(op.join(directory, f"ROC_{save_title}.svg"))


def display_precision_recall_curve(
    X_test, y_test, model, save_title=None, directory="images"
):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    from sklearn.metrics import PrecisionRecallDisplay


    y_proba = model.predict_proba(X_test)
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test.shape[1]

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_proba[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_proba[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve( y_test.ravel(), y_proba.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_proba, average="micro")

    # Display
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    plt.show()
    plt.savefig(op.join(directory, f"precision_recall_curve_{save_title}.svg"))


##############################
def plot_delta_time(y_true_list, y_pred_list):
    """Visualisation tool."""
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    delta_loc = y_true[:, 0] - y_pred[:, 0]
    delta_roc = y_true[:, 1] - y_pred[:, 1]
    df = {"loc": delta_loc, "roc": delta_roc}
    df = pd.DataFrame(data=df)

    #     sns.set(style="darkgrid")

    sns.histplot(data=df, x="loc", color="red", kde=True, label="loc")
    sns.histplot(data=df, x="roc", color="blue", kde=True, label="roc")

    plt.title("Delta time distribution")
    plt.legend()
    plt.show()


def multiplot_delta_time(
    y_true_list,
    y_pred_list,
    iloc,
    iroc,
    xlabel="Distribution of the error prediction for the LOC ",
    xlabel_2="Distribution of the error prediction for the ROC ",
    confidence_interval=True,
    save_title=None,
    directory="images",
):
    """Visualisation tool."""
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    delta_loc = y_true[:, 0] - y_pred[:, 0]
    delta_roc = y_true[:, 1] - y_pred[:, 1]

    df = {"loc": delta_loc, "roc": delta_roc}
    df = pd.DataFrame(data=df)

    #     sns.set(style="darkgrid")

    fig, (ax_box, ax_hist) = plt.subplots(
        2,
        2,
        figsize=(15, 5),
        sharex=False,
        gridspec_kw={"height_ratios": (0.5, 0.5)},
    )

    sns.boxplot(
        df["loc"],
        medianprops={"color": "lightblue"},
        showmeans=True,
        meanprops={"markerfacecolor": "white"},
        ax=ax_box[0],
    )
    ax_kde = sns.histplot(
        data=df, x="loc", color="red", kde=True, ax=ax_hist[0], bins=58
    )
    ax_kde.set_xlabel("Delta time (in sec)")
    ax_kde.set_ylabel("Number of patients")
    kde_x, kde_y = ax_kde.lines[0].get_data()

    if confidence_interval:
        ax_kde.fill_between(
            kde_x,
            kde_y,
            where=(kde_x < iloc[0]) | (kde_x > iloc[1]),
            interpolate=True,
            color="blue",
            alpha=0.4,
        )
        # annotate
        x_text_ic_low = kde_x[len(kde_x < iloc[0]) // 2]
        x_text_ic_high = len(kde_x[kde_x > iloc[1]]) // 2 + len(kde_x[kde_x < iloc[1]])
        x_ic_low = kde_x[kde_x < iloc[0]][-1]
        x_ic_high = kde_x[kde_x < iloc[1]][-1]
        y_arrow = kde_y[kde_x < iloc[0]][0]

        ax_kde.annotate(
            "CI 95%",
            xy=(x_ic_low, y_arrow),
            xytext=(x_text_ic_low, 8),
            arrowprops=dict(arrowstyle="-", color="blue", alpha=0.5),
        )
        ax_kde.annotate(
            "CI 95%",
            xy=(x_ic_high, y_arrow),
            xytext=(x_text_ic_high, 8),
            arrowprops=dict(arrowstyle="-", color="blue", alpha=0.5),
        )
    else:
        q1, q3 = np.percentile(delta_loc, [25, 75])
        print("loc : IQR 1 ", q1, " IQR 3 ", q3)
        ax_kde.fill_between(
            kde_x,
            kde_y,
            where=(kde_x < q1) | (kde_x > q3),
            interpolate=True,
            color="blue",
            alpha=0.4,
        )

    sns.boxplot(
        df["roc"],
        medianprops={"color": "lightblue"},
        showmeans=True,
        meanprops={"markerfacecolor": "white"},
        ax=ax_box[1],
    )
    ax_kde = sns.histplot(data=df, x="roc", color="blue", kde=True, ax=ax_hist[1])
    ax_kde.set_xlabel("Delta time (in sec)")
    ax_kde.set_ylabel("Number of patients")
    kde_x, kde_y = ax_kde.lines[0].get_data()

    if confidence_interval:
        ax_kde.fill_between(
            kde_x,
            kde_y,
            where=(kde_x < iroc[0]) | (kde_x > iroc[1]),
            interpolate=True,
            color="red",
            alpha=0.4,
        )

        # annotate
        x_text_ic_low = kde_x[len(kde_x < iroc[0]) // 2]
        x_text_ic_high = len(kde_x[kde_x > iroc[1]]) // 2 + len(kde_x[kde_x < iroc[1]])
        x_ic_low = kde_x[kde_x < iroc[0]][-1]
        x_ic_high = kde_x[kde_x < iroc[1]][-1]
        y_arrow = kde_y[kde_x < iroc[0]][0]

        ax_kde.annotate(
            "CI 95%",
            xy=(x_ic_low, y_arrow),
            xytext=(x_text_ic_low, 8),
            arrowprops=dict(arrowstyle="-", color="r", alpha=0.5),
        )
        ax_kde.annotate(
            "CI 95%",
            xy=(x_ic_high, y_arrow),
            xytext=(x_text_ic_high, 8),
            arrowprops=dict(arrowstyle="-", color="r", alpha=0.5),
        )
    else:
        q1, q3 = np.percentile(delta_roc, [25, 75])
        print("roc : IQR 1 ", q1, " IQR 3 ", q3)

        ax_kde.fill_between(
            kde_x,
            kde_y,
            where=(kde_x < q1) | (kde_x > q3),
            interpolate=True,
            color="red",
            alpha=0.4,
        )

    ax_box[1].set(title=xlabel_2, xlabel="")
    ax_box[0].set(title=xlabel, xlabel="")

    plt.savefig(op.join(directory, f"distribution_{save_title}.svg"))
