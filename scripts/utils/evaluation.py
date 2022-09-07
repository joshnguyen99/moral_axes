import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
import scipy.stats

# False positive rate values for interpolation
FPR_RANGE = np.linspace(0, 1, 100)
# Recall values for interpolation
RECALL_RANGE = np.linspace(0, 1, 100)


def evaluate_binary_scorer_fold(y_true, y_pred, y_score):
    """
    Evaluate the predictions of a scored binary classifier on one fold.
    :param y_true: Ground-truth labels (0 or 1) of N elements.
    :param y_pred: Binary predictions (0 or 1) of N elements.
    :param y_score: Prediction scores (higher scores = higher degree that an instance
    is of class 1) of N elements.
    :return: A dictionary of evaluation metrics
    """
    results = dict()

    # Binary classification metrics
    results["accuracy"] = accuracy_score(y_true=y_true, y_pred=y_pred)
    results["precision"] = precision_score(y_true=y_true, y_pred=y_pred)
    results["recall"] = recall_score(y_true=y_true, y_pred=y_pred)
    results["f1"] = f1_score(y_true=y_true, y_pred=y_pred)

    # Ranked retrieval metrics
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true=y_true, y_score=y_score)
    auc_roc = auc(x=fpr, y=tpr)
    # Linearly interpolate the TPR rate at each FPR value
    interpolated_tpr = np.interp(x=FPR_RANGE, xp=fpr, fp=tpr)
    interpolated_tpr[-1] = 1.0
    results["roc_curve"] = dict(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds_roc,
        auc=auc_roc,
        interpolated_fpr=FPR_RANGE,
        interpolated_tpr=interpolated_tpr
    )

    # Precision-recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    recall = recall[::-1]
    precision = precision[::-1]
    thresholds_pr = thresholds_pr[::-1]
    auc_pr = auc(x=recall, y=precision)
    # Linearly interpolate the precision at each recall value
    interpolated_precision = np.interp(x=RECALL_RANGE, xp=recall, fp=precision)
    interpolated_precision[0] = 1.0
    results["pr_curve"] = dict(
        recall=recall,
        precision=precision,
        thresholds=thresholds_pr,
        auc=auc_pr,
        interpolated_recall=RECALL_RANGE,
        interpolated_precision=interpolated_precision
    )

    return results


def mean_sd_ci(values, confidence_level=0.95):
    n = len(values)
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0)
    sd_lower, sd_upper = mean - sd, mean + sd
    se = scipy.stats.sem(values, axis=0)
    ci_lower, ci_upper = scipy.stats.t.interval(confidence_level,
                                                df=n - 1,
                                                loc=mean,
                                                scale=se)
    return dict(
        mean=mean,
        sd_lower=sd_lower,
        sd_upper=sd_upper,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


def evaluate_binary_scorer(folds,
                           tpr_confidence=0.95,
                           auc_roc_confidence=0.95,
                           precision_confidence=0.95,
                           auc_pr_confidence=0.95):
    """
    Evaluate the predictions of a scored binary classifier on k folds.
    :param folds: A list of dictionaries in the form {"y_true": [], "y_pred": [], "y_score": []}. Each
    dictionary corresponds to the prediction results for a fold.
    :param tpr_confidence: Confidence level for true positive rates on the ROC curve
    :param auc_roc_confidence: Confidence level for the AUC of the ROC curve
    :param precision_confidence: Confidence level for the precision on the PR curve
    :param auc_pr_confidence: Confidence level for the AUC of the PR curve
    :return: A dictionary of evaluation metrics
    """
    results = dict(
        fold_results=[],
        aggregate=dict(
            roc_curve=dict(),
            pr_curve=dict()
        ),
    )
    for fold in folds:
        y_true = fold["y_true"]
        y_pred = fold["y_pred"]
        y_score = fold["y_score"]
        fold_results = evaluate_binary_scorer_fold(y_true=y_true,
                                                   y_pred=y_pred,
                                                   y_score=y_score)
        results["fold_results"].append(fold_results)

    # Aggregate the interpolated ROC curves
    all_tprs = np.array([r["roc_curve"]["interpolated_tpr"] for r in results["fold_results"]])
    all_auc_roc = np.array([r["roc_curve"]["auc"] for r in results["fold_results"]])
    agg_tprs = mean_sd_ci(values=all_tprs, confidence_level=tpr_confidence)
    agg_auc_roc = mean_sd_ci(values=all_auc_roc, confidence_level=auc_roc_confidence)
    results["aggregate"]["roc_curve"] = dict(
        fpr=results["fold_results"][0]["roc_curve"]["interpolated_fpr"],
        tpr=agg_tprs["mean"],
        tpr_sd_lower=agg_tprs["sd_lower"],
        tpr_sd_upper=agg_tprs["sd_upper"],
        tpr_ci_lower=agg_tprs["ci_lower"],
        tpr_ci_upper=agg_tprs["ci_upper"],
        auc=agg_auc_roc["mean"],
        auc_sd=agg_auc_roc["mean"] - agg_auc_roc["sd_lower"],
        auc_ci=(agg_auc_roc["ci_lower"], agg_auc_roc["ci_upper"])
    )

    # Aggregate the interpolated PR curves
    all_precision = np.array([r["pr_curve"]["interpolated_precision"] for r in results["fold_results"]])
    all_auc_pr = np.array([r["pr_curve"]["auc"] for r in results["fold_results"]])
    agg_precision = mean_sd_ci(values=all_precision, confidence_level=precision_confidence)
    agg_auc_pr = mean_sd_ci(values=all_auc_pr, confidence_level=auc_pr_confidence)
    results["aggregate"]["pr_curve"] = dict(
        recall=results["fold_results"][0]["pr_curve"]["interpolated_recall"],
        precision=agg_precision["mean"],
        precision_sd_lower=agg_precision["sd_lower"],
        precision_sd_upper=agg_precision["sd_upper"],
        precision_ci_lower=agg_precision["ci_lower"],
        precision_ci_upper=agg_precision["ci_upper"],
        auc=agg_auc_pr["mean"],
        auc_sd=agg_auc_pr["mean"] - agg_auc_pr["sd_lower"],
        auc_ci=(agg_auc_pr["ci_lower"], agg_auc_pr["ci_upper"])
    )

    return results
