import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import pickle
import json
import argparse

FOUNDATIONS = ["authority", "care", "fairness", "loyalty", "sanctity"]
DATA_PATH = "data/sentence_mf_one_v_all.csv"
SENTENCE_ROBERTA_EMB_PATH = "data/embeddings/sentence_roberta.npz"
GLOVE_EMB_PATH = "data/embeddings/glove_twitter_200.npz"


def prepare_data(foundation, args):

    # TODO: adapt to MFTC and MFRC datasets

    if foundation not in FOUNDATIONS:
        raise ValueError("Invalid foundation: {}".format(foundation))

    # Load data
    dataset = pd.read_csv(DATA_PATH, index_col=0)
    dataset.drop(columns=[], inplace=True)

    # Only keep sentences that have been seen by the annotator
    # Fold = 0: test set
    # Fold = -1: not seen
    # Fold = 1,..., 10: training set
    dataset_train = dataset[dataset[f"{foundation}_fold"] > 0]
    dataset_test = dataset[dataset[f"{foundation}_fold"] == 0]

    train_indices = dataset_train.index
    fold_ids = dataset_train[f"{foundation}_fold"]
    train_sentences = dataset_train["sentence"].values
    y_train = dataset_train[f"{foundation}_label"].values

    test_indices = dataset_test.index
    test_sentences = dataset_test["sentence"].values
    y_test = dataset_test[f"{foundation}_label"].values

    # Load embeddings
    # TODO: add more embeddings (bow, tfidf, spacy, etc.)
    if args.embedding == "sentence_roberta":
        emb_path = SENTENCE_ROBERTA_EMB_PATH
    elif args.embedding == "glove":
        emb_path = GLOVE_EMB_PATH
    X = np.load(emb_path)["arr_0"]

    assert X.shape[0] == dataset.shape[0]

    X_train = X[train_indices]
    X_test = X[test_indices]

    print("Training set size: {}".format(X_train.shape[0]))
    print("Test set size    : {}".format(X_test.shape[0]))
    print("Training set's positive proportion: {:.2f}%".format(np.mean(y_train) * 100))
    print("Test set's positive proportion    : {:.2f}%\n".format(np.mean(y_test) * 100))

    return X_train, X_test, y_train, y_test, fold_ids


def train_cv(X_train, y_train, fold_ids):
    results = []
    C_range = [1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    cv_results = [[] for _ in C_range]
    fold_ids_unique, valid_fold_sizes = np.unique(fold_ids, return_counts=True)

    for fold_id in fold_ids_unique:
        print("\tPerforming cross validation for fold {}".format(fold_id))
        # mask = 0 => validation set
        # mask = -1 => training set
        train_valid_mask = np.array([0 if fold == fold_id else -1 for fold in fold_ids])

        X_train_fold = X_train[train_valid_mask == -1]
        y_train_fold = y_train[train_valid_mask == -1]
        X_valid_fold = X_train[train_valid_mask == 0]
        y_valid_fold = y_train[train_valid_mask == 0]

        print("\t\tTraining set size  : {}".format(X_train_fold.shape[0]))
        print("\t\tValidation set size: {}".format(X_valid_fold.shape[0]))
        print("\t\tTraining set's positive proportion  : {:.2f}%".format(np.mean(y_train_fold) * 100))
        print("\t\tValidation set's positive proportion: {:.2f}%".format(np.mean(y_valid_fold) * 100))

        cv = PredefinedSplit(test_fold=train_valid_mask)
        cv.get_n_splits(X_train, y_train)

        # Find the best parameters, based on the AUROC on the validation set
        model = GridSearchCV(LogisticRegression(max_iter=10000, tol=1e-10,
                                                solver="lbfgs", penalty="l2"),
                             param_grid={"C": C_range},
                             cv=cv,
                             n_jobs=-1,
                             scoring=make_scorer(roc_auc_score),
                             refit=False)
        model.fit(X_train, y_train)
        print("\t\tBest C    : {:.2e}".format(model.best_params_["C"]))
        print("\t\tBest AUROC: {:.4f}".format(model.best_score_))

        # Get the score for each C value
        scores = model.cv_results_["split0_test_score"]
        for r, score in zip(cv_results, scores):
            r.append(score)

        # Get the best model and train it on the training set
        best_params = model.best_params_
        best_model = LogisticRegression(max_iter=10000, tol=1e-10,
                                        solver="lbfgs", penalty="l2",
                                        n_jobs=-1,
                                        **best_params)
        best_model.fit(X_train_fold, y_train_fold)

        # Evaluate the model on the validation set
        y_valid_score = best_model.predict_proba(X_valid_fold)[:, 1]
        y_valid_score_median = np.median(y_valid_score)
        y_valid_pred = (y_valid_score >= y_valid_score_median).astype(int)

        # Store predictions on the validation set
        result = {
            "y_true": [int(y) for y in y_valid_fold],
            "y_score": [float(score) for score in y_valid_score],
            "y_pred": [int(y) for y in y_valid_pred],
        }
        results.append(result)

    # Compute the average score for each C value
    cv_results_mean = [np.average(r, weights=valid_fold_sizes) for r in cv_results]
    # Get the best C across folds
    best_C = C_range[np.argmax(cv_results_mean)]
    print()
    print("\tBest C    : {:.2e}".format(best_C))
    print("\tBest AUROC: {:.4f}".format(np.max(cv_results_mean)))
    # Train the final model on the whole training set
    final_model = LogisticRegression(max_iter=10000, tol=1e-10,
                                     solver="lbfgs", penalty="l2",
                                     C=best_C, n_jobs=-1)
    final_model.fit(X_train, y_train)

    return results, final_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train logistic regression on MFD dataset.')
    parser.add_argument("-e", "--embedding", type=str, required=True,
                        help="Name of the embedding (sentence_roberta or glove)",
                        choices=["sentence_roberta", "glove"])
    args = parser.parse_args()

    print("USING {} EMBEDDING".format(args.embedding.upper()))

    for foundation in FOUNDATIONS:
        print("TRAINING BINARY CLASSIFIER FOR FOUNDATION: {}\n".format(foundation).upper())
        X_train, X_test, y_train, y_test, fold_ids = prepare_data(foundation, args)
        cv_results, final_model = train_cv(X_train, y_train, fold_ids)

        with open(f"data/sentence_classifiers/logreg_{args.embedding}_{foundation}.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(f"data/mfd_scoring_results/by_foundation/logreg_{args.embedding}_{foundation}_train.json", "w") as f:
            json.dump(cv_results, f)

        # Evaluate on the test set
        y_test_score = final_model.predict_proba(X_test)[:, 1]
        y_test_score_median = np.median(y_test_score)
        y_test_pred = (y_test_score >= y_test_score_median).astype(int)
        test_result = {
            "y_true": [int(y) for y in y_test],
            "y_score": [float(score) for score in y_test_score],
            "y_pred": [int(y) for y in y_test_pred],
        }

        with open(f"data/mfd_scoring_results/by_foundation/logreg_{args.embedding}_{foundation}_test.json", "w") as f:
            json.dump(test_result, f)

        print("=" * 80)
