"""
Finding thresholds for binary classifiers.
"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
)


def threshold_max_f1(y_true, y_score):
    """
    Find the threshold that maximizes the F1 score.

    Args:
        y_true: A list of ground truth labels.
        y_score: A list of predicted scores.

    Returns:
        The threshold that maximizes the F1 score.

    """
    precision, recall, thresholds = precision_recall_curve(y_true=y_true,
                                                           probas_pred=y_score)
    f1 = 2 * (precision * recall) / (precision + recall)
    return thresholds[np.argmax(f1)]


def threshold_max_gmean(y_true, y_score):
    """
    Find the threshold that maximizes the geometric mean of sensitivity and
    specificity, i.e., sqrt(sentivity * specificity).

    Args:
        y_true: A list of ground truth labels.
        y_score: A list of predicted scores.

    Returns:
        The threshold that maximizes the geometric mean.
    """

    fpr, tpr, thresholds_roc = roc_curve(y_true=y_true, y_score=y_score)
    sensitivity = tpr
    specificity = 1 - fpr
    gmeans = np.sqrt(sensitivity * specificity)
    return thresholds_roc[np.argmax(gmeans)]


def threshold_max_diff_tpr_fpr(y_true, y_score):
    """
    Find the threshold that maximizes the difference between true positive rate
    and false positive rate.

    Args:
        y_true: A list of ground truth labels.
        y_score: A list of predicted scores.

    Returns:
        The threshold that maximizes the difference between true positive rate
        and false positive rate.

    """
    fpr, tpr, thresholds_roc = roc_curve(y_true=y_true, y_score=y_score)
    return thresholds_roc[np.argmax(tpr - fpr)]


def threshold_top_x(y_score, x):
    """
    Set the threshold to the xth percentile of the predicted scores.

    Args:
        y_score: A list of predicted scores.
        x: The top-x accuracy, in [0, 100].

    Returns:
        The threshold that maximizes the top-x accuracy.
    """
    assert 0 <= x and x <= 100
    threshold = np.percentile(y_score, q=x)
    return threshold
