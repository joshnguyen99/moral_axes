"""
Scripts for binary clasifier calibration.
"""
import numpy as np


def calibrate_saerens(y_test_score, y_train):
    """
    Calibrate the test scores of a binary classifier using
    binary labels in the training set. The formula is
        p = p / (p + f_np * (1 - p))
    where p is the score of a test example and f_np is the 
    ratio between negative and positive examples in the training set.

    Source: Marco Saerens, Patrice Latinne, Christine Decaestecker;
            Adjusting the Outputs of a Classifier to New a Priori Probabilities:
            A Simple Procedure. Neural Comput 2002; 14 (1): 21–41. 
            doi: https://doi.org/10.1162/089976602753284446

    Args:
        y_test_score: predicted scores for test examples. Array of float. 
        y_train: binary labels for training examples. Array of {0, 1}.

    Returns:
        Calibrated scores for test examples. Array or float, depending on
        the type of y_test_score.
    """

    # Proportion of positive examples in the training set
    train_pos_prop = np.mean(y_train)

    # Ratio between negative and positive examples in the training set
    neg_to_pos_prop = (1 - train_pos_prop) / train_pos_prop

    # Calibrate the test scores
    y_test_score_calibrated = \
        y_test_score / (y_test_score + neg_to_pos_prop * (1 - y_test_score))

    return y_test_score_calibrated


def inverse_calibrate_saerens(y_test_score_calibrated, y_train):
    """
    Apply the inverse of the Saerens calibration function.

    Args:
        y_test_score_calibrated: calibrated scores for test examples.
                                 Array of float.
        y_train: binary labels for training examples. Array of {0, 1}.

    Returns:
        The original scores for test examples. Array or float, depending on
        the type of y_test_score_calibrated.
    """

    # Proportion of positive examples in the training set
    train_pos_prop = np.mean(y_train)

    # Ratio between negative and positive examples in the training set
    neg_to_pos_prop = (1 - train_pos_prop) / train_pos_prop

    # Apply inverse calibration
    y_test_score = y_test_score_calibrated * neg_to_pos_prop / \
        (1 - y_test_score_calibrated + y_test_score_calibrated * neg_to_pos_prop)

    return y_test_score
