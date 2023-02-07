"""
Scripts for binary clasifier calibration.
"""
import numpy as np
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    """
    Base class for calibrators. Each calibrator should implement
    the method predict to transform the scores by a binary classifier.
    """

    def __init__(self):
        pass

    def predict(self, T):
        pass


class CalibratorSaerens(Calibrator):
    def __init__(self, neg_to_pos_prop):
        super().__init__()
        self.neg_to_pos_prop = neg_to_pos_prop

    def predict(self, T):
        y_score_calibrated = T / (T + self.neg_to_pos_prop * (1 - T))
        return y_score_calibrated


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
        Calibration model. This can be called for new test scores.
    """

    # Proportion of positive examples in the training set
    train_pos_prop = np.mean(y_train)

    # Ratio between negative and positive examples in the training set
    neg_to_pos_prop = (1 - train_pos_prop) / train_pos_prop

    # Calibrate the test scores
    model = CalibratorSaerens(neg_to_pos_prop)
    y_test_score_calibrated = model.predict(T=y_test_score)

    return y_test_score_calibrated, model


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


def calibrate_platt(y_score, y_true):
    """
    Calibrate the scores of a binary classifier using Platt scaling.
    This is also called sigmoid calibration.
    Given a score p, the calibrated score is
        p_calibrated = 1 / (1 + exp(a * p + b))
    where the parameters a and b are found by maximum likelihood.
    Platt scaling should be used for a held out dataset, as using it
    on the test set introduces bias.

    Args:
        y_true: groud-truth labels.
        y_score: predicted scores.

    Returns:
        Calibrated scores.
        Calibration model of type _SigmoidCalibration, consisting of 
            slope a and intercept b.
    """

    model = _SigmoidCalibration()
    model.fit(X=y_score, y=y_true)
    return model.predict(T=y_score), model


def inverse_calibrate_platt(y_score_calibrated, model: _SigmoidCalibration):
    """
    Apply the inverse of the Platt calibration function.

    Args:
        y_score_calibrated: calibrated scores.
        model: calibration model of type _SigmoidCalibration, consisting of 
            slope a and intercept b.

    Returns:
        The original scores.
    """

    a, b = model.a_, model.b_
    y_score = (np.log(1 - y_score_calibrated) - np.log(y_score_calibrated) - b) / a
    return y_score


def calibrate_isotonic(y_score, y_true):
    """
    Calibrate the scores of a binary classifier using isotonic regression.

    Args:
        y_true: groud-truth labels.
        y_score: predicted scores.

    Returns:
        Calibrated scores.
        Calibration model of type IsotonicRegression.
    """

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(X=y_score, y=y_true)
    return model.predict(T=y_score), model
