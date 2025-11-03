"""Evaluation of the model."""

import os.path as op

import numpy as np
import pandas as pd
from eeg_features import compute_accuracy
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams
from prediction import predict_loc_roc, predict_probabilities
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm

# import shap
from boostlocroc.utils.data import get_subject

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


def evaluate(
    y_true,
    y_pred,
    precision,
    subject_name,
    verbose=True,
    display_pred=False,
    parquet=None,
    raw=None,
    vspan_predictions=False,
    zoom=False,
    save=False,
    directory=None,
    save_title=None,
):
    """Choose the Option display : pred ou eeg ou Nada."""
    loc, roc = y_true
    pred_loc, pred_roc = y_pred

    loc_score = compute_accuracy(loc, pred_loc, precision)
    roc_score = compute_accuracy(roc, pred_roc, precision)

    delta_loc = pred_loc - loc
    delta_roc = pred_roc - roc

    if verbose:
        print("y_true ", y_true, "y_pred ", y_pred)
        print("scores:", f"loc = {loc_score}", f"roc = {roc_score}", sep="\n -> ")
        print(
            "delta time (sec.):",
            f"loc = {delta_loc}",
            f"roc = {delta_roc}",
            sep="\n -> ",
        )

    return (loc_score, roc_score), (delta_loc, delta_roc)


def metrics(
    y_true, y_pred, score_loc, score_roc, delta_loc, delta_roc, confidence=0.95
):
    """Get a table with the wanted metrics."""
    y_true = list(map(list, zip(*y_true)))
    y_pred = list(map(list, zip(*y_pred)))

    print(
        "mean square error (in min) loc=> ",
        np.sqrt(mean_squared_error(y_true[0], y_pred[0])) / 60,
    )
    print(
        "mean square error (in min) roc=> ",
        np.sqrt(mean_squared_error(y_true[1], y_pred[1])) / 60,
    )

    delta_loc = np.array(delta_loc)
    delta_roc = np.array(delta_roc)

    print("mean score loc => ", np.mean(score_loc))
    print("mean score roc => ", np.mean(score_roc))
    print("median score loc => ", np.median(score_loc))
    print("median score roc => ", np.median(score_roc))
    print("--------------------")
    mean_loc = np.mean(delta_loc)
    mean_roc = np.mean(delta_roc)
    median_loc = np.median(delta_loc)
    median_roc = np.median(delta_roc)

    print("mean delta loc", mean_loc)
    print("mean delta roc", mean_roc)
    print("median delta loc", median_loc)
    print("median delta roc", median_roc)
    print("--------------------")
    print("mean delta positive loc => ", np.mean(delta_loc[delta_loc > 0]))
    print("mean delta negative loc => ", np.mean(delta_loc[delta_loc < 0]))
    print("median delta positive loc => ", np.median(delta_loc[delta_loc > 0]))
    print("median delta negative loc => ", np.median(delta_loc[delta_loc < 0]))
    print("mean delta negative roc => ", np.mean(delta_roc[delta_roc < 0]))
    print("mean delta positive roc => ", np.mean(delta_roc[delta_roc > 0]))
    print("median delta positive roc => ", np.median(delta_roc[delta_roc > 0]))
    print("median delta negative roc => ", np.median(delta_roc[delta_roc < 0]))

    std_loc = np.std(delta_loc)
    std_roc = np.std(delta_roc)
    delta_length = len(delta_loc)

    print("std loc => ", std_loc)
    print("std roc => ", std_roc)

    t_crit = np.abs(t.ppf((1 - confidence) / 2, delta_length - 1))

    # mean - x where x = std_loc * t_crit / np.sqrt(len(delta))
    x_loc = std_loc * t_crit / np.sqrt(delta_length)
    ic_loc = median_loc - x_loc, median_loc + x_loc

    x_roc = std_roc * t_crit / np.sqrt(delta_length)
    ic_roc = median_roc - x_roc, median_roc + x_roc

    print("confidence interval delta_loc => ", ic_loc)
    print("confidence interval delta_roc => ", ic_roc)

    return (
        ic_loc,
        ic_roc,
        (std_loc, std_roc),
        (mean_loc, median_loc),
        (mean_roc, median_roc),
        (delta_loc, delta_roc),
        (y_true, y_pred),
    )


def evaluate_one_model(
    idx_test,
    X,
    Y,
    weights_path,
    loc_roc_labels,
    precision,
    clean_operation=False,
    epochs_duration=30,
    vspan_predictions=False,
    display_pred=True,
    verbose=False,
):
    """Evaluate the model."""
    X.drop(["epochs"], axis=1, inplace=True)
    test_input = X.iloc[idx_test]

    scores_loc = []
    scores_roc = []

    deltas_loc = []
    deltas_roc = []

    y_true_list = []
    y_pred_list = []

    really_bad_loc = []
    really_bad_roc = []

    for idx in tqdm(test_input.index.unique()):
        X_subject_name, Y_subject, parquet, raw, y_true = get_subject(
            X, Y, idx, loc_roc_labels, clean_operation=clean_operation
        )
        y_pred = predict_loc_roc(
            X_subject_name, weights_path, epochs_duration=epochs_duration
        )

        if verbose:
            print(idx, end=" ==>\n")
            print("labels", y_true)
            print("preds", y_pred)

        (loc_score, roc_score), (delta_loc, delta_roc) = evaluate(
            y_true,
            y_pred,
            precision,
            idx,
            verbose=False,
            display_pred=display_pred,
            parquet=parquet,
            raw=raw,
            vspan_predictions=vspan_predictions,
        )

        y_pred_list.append(y_pred)

        scores_loc.append(loc_score)
        scores_roc.append(roc_score)
        deltas_loc.append(delta_loc)
        deltas_roc.append(delta_roc)
        y_true_list.append(y_true)

        if abs(delta_loc) > 120:
            really_bad_loc.append(idx)
        if abs(delta_roc) > 120:
            really_bad_roc.append(idx)

        if verbose:
            print("loc_score", loc_score)
            print("roc_score", roc_score)
            print("delta_loc", delta_loc)
            print("delta_roc", delta_roc, end="\n-----\n")

    return (
        (scores_loc, scores_roc, deltas_loc, deltas_roc, y_true_list),
        y_pred_list,
        (really_bad_loc, really_bad_roc),
    )


# NEW OSA


def get_summary_stat(
    y_true_list, y_pred_list, accuracy=None, save_title=None, directory=None
):
    """Get summary statistics."""
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    delta_loc = y_true[:, 0] - y_pred[:, 0]  # delta + => prediction is before
    delta_roc = y_true[:, 1] - y_pred[:, 1]

    # sample size > 30, we can approximate the student's distribution with a normal distribution thanks to the central limit theorem
    confidence = 0.95
    delta_len = len(delta_loc)
    t_crit = np.abs(
        t.ppf((1 - confidence) / 2, delta_len - 1)
    )  # .ppf = the inverse cumulative distribution

    # LOC
    mu_loc = delta_loc.mean()
    med_loc = np.median(delta_loc)
    std_loc = delta_loc.std()
    ic_loc_low, ic_loc_high = (
        mu_loc - std_loc * t_crit / np.sqrt(delta_len),
        mu_loc + std_loc * t_crit / np.sqrt(delta_len),
    )
    mae_loc = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    rmse_loc = mean_squared_error(y_true[0], y_pred[0], squared=False)
    q1_l, q3_l = np.percentile(delta_loc, [25, 75])

    # ROC
    mu_roc = delta_roc.mean()
    med_roc = np.median(delta_roc)
    std_roc = delta_roc.std()
    ic_roc_low, ic_roc_high = (
        mu_roc - std_roc * t_crit / np.sqrt(len(delta_roc)),
        mu_roc + std_roc * t_crit / np.sqrt(len(delta_roc)),
    )
    mae_roc = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    rmse_roc = mean_squared_error(y_true[1], y_pred[1], squared=False)
    q1_r, q3_r = np.percentile(delta_roc, [25, 75])

    # Export to csv
    metrics_loc = (
        mu_loc,
        med_loc,
        std_loc,
        ic_loc_low,
        ic_loc_high,
        q1_l,
        q3_l,
        mae_loc,
        rmse_loc,
        accuracy,
    )
    metrics_roc = (
        mu_roc,
        med_roc,
        std_roc,
        ic_roc_low,
        ic_roc_high,
        q1_r,
        q3_r,
        mae_roc,
        rmse_roc,
    )

    metrics_df = pd.concat(
        [
            pd.DataFrame(
                metrics_loc,
                index=[
                    "Mean",
                    "Median",
                    "sdt",
                    "CI 95% (low)",
                    "CI 95% (high)",
                    "Q1",
                    "Q3",
                    "MAE",
                    "RMSE",
                    "AccuracyY",
                ],
                columns=["LOC"],
            ),
            pd.DataFrame(
                metrics_roc,
                index=[
                    "Mean",
                    "Median",
                    "sdt",
                    "CI 95% (low)",
                    "CI 95% (high)",
                    "Q1",
                    "Q3",
                    "MAE",
                    "RMSE",
                ],
                columns=["ROC"],
            ),
        ],
        axis=1,
    )

    metrics_df.to_csv(op.join(directory, f"metrics_{save_title}.csv"))

    return metrics_df, (ic_loc_low, ic_loc_high), (ic_roc_low, ic_roc_high)


def new_summary_stat(
    y_true_list, y_pred_list, accuracy=None, save_title=None, directory=None
):
    """Get summary statistics."""
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    delta_loc = y_true[:, 0] - y_pred[:, 0]  # delta + => prediction is before
    delta_roc = y_true[:, 1] - y_pred[:, 1]

    # sample size > 30, we can approximate the student's distribution with a normal distribution thanks to the central limit theorem
    confidence = 0.95
    delta_len = len(delta_loc)
    t_crit = np.abs(
        t.ppf((1 - confidence) / 2, delta_len - 1)
    )  # .ppf = the inverse cumulative distribution

    # LOC
    mu_loc = delta_loc.mean()
    med_loc = np.median(delta_loc)
    std_loc = delta_loc.std()
    ic_loc_low, ic_loc_high = (
        mu_loc - std_loc * t_crit / np.sqrt(delta_len),
        mu_loc + std_loc * t_crit / np.sqrt(delta_len),
    )
    mae_loc = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    rmse_loc = mean_squared_error(y_true[0], y_pred[0], squared=False)
    q1_l, q3_l = np.percentile(delta_loc, [25, 75])

    # ROC
    mu_roc = delta_roc.mean()
    med_roc = np.median(delta_roc)
    std_roc = delta_roc.std()
    ic_roc_low, ic_roc_high = (
        mu_roc - std_roc * t_crit / np.sqrt(len(delta_roc)),
        mu_roc + std_roc * t_crit / np.sqrt(len(delta_roc)),
    )
    mae_roc = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    rmse_roc = mean_squared_error(y_true[1], y_pred[1], squared=False)
    q1_r, q3_r = np.percentile(delta_roc, [25, 75])

    # Export to csv
    metrics_loc = (
        mu_loc,
        med_loc,
        std_loc,
        ic_loc_low,
        ic_loc_high,
        q1_l,
        q3_l,
        mae_loc,
        rmse_loc,
        accuracy,
    )
    metrics_roc = (
        mu_roc,
        med_roc,
        std_roc,
        ic_roc_low,
        ic_roc_high,
        q1_r,
        q3_r,
        mae_roc,
        rmse_roc,
    )

    return (
        (
            std_loc,
            std_roc,
            q1_l,
            q3_l,
            q1_r,
            q3_r,
            (ic_loc_low, ic_loc_high),
            (ic_roc_low, ic_roc_high),
        ),
        metrics_loc,
        metrics_roc,
    )


def new_evaluate_model(
    X,
    Y,
    model,
    loc_roc_labels,
    epochs_duration=30,
    clean_operation=False,
    min_max_norm=False,
):
    """Evaluate the model."""
    y_true_list = []
    y_pred_list = []

    for idx in tqdm(X.index.unique()):
        X_subject_name, Y_subject, y_true = get_subject(X, Y, idx, loc_roc_labels)
        probability = predict_probabilities(
            model, X_subject_name, min_max_norm=min_max_norm
        )
        y_pred = predict_loc_roc(probability, epochs_duration=epochs_duration)

        y_pred_list.append(y_pred)
        y_true_list.append(y_true)

    y_true_f = np.array(y_true_list)
    y_pred_f = np.array(y_pred_list)

    delta_loc = y_true_f[:, 0] - y_pred_f[:, 0]  # delta + => prediction is before
    delta_roc = y_true_f[:, 1] - y_pred_f[:, 1]

    return (y_true_list, y_pred_list), (delta_loc, delta_roc)


def shap_values(model, X_test, save_title=None, directory=None):
    """SHAP function."""
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)

    shap.summary_plot(
        shap_values,
        X_test.values,
        plot_type="bar",
        class_names=["Zero", "One", "Two"],
        feature_names=X_test.columns,
        show=False,
    )
    plt.savefig(op.join(directory, f"shap_summary_plot_{save_title}.svg"))

    shap.summary_plot(
        shap_values[0], X_test.values, feature_names=X_test.columns, show=False
    )
    plt.savefig(op.join(directory, f"shap_summary_0_{save_title}.svg"))
    shap.summary_plot(
        shap_values[1], X_test.values, feature_names=X_test.columns, show=False
    )
    plt.savefig(op.join(directory, f"shap_summary_1_{save_title}.svg"))
    shap.summary_plot(
        shap_values[2], X_test.values, feature_names=X_test.columns, show=False
    )
    plt.savefig(op.join(directory, f"shap_summary_2_{save_title}.svg"))

    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(op.join(directory, f"shap_beeswarm_{save_title}.png"))

    return shap_values, explainer
