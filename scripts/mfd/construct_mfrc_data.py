import pandas as pd
import spacy
from tqdm import tqdm
import argparse
import os
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

nlp = spacy.load("en_core_web_md", disable=["ner", "senter"])


def parse_args():
    parser = argparse.ArgumentParser(description='Create MFRC data')
    parser.add_argument("--raw_data_path",
                        type=str,
                        help="Path to the MFRC data from Huggingface.",
                        default="data/final_mfrc_data.csv")
    parser.add_argument("--output_path",
                        type=str,
                        help="Path to the output file. In .csv format.",
                        default="data/mfrc_preprocessed.csv",
                        )
    parser.add_argument("--n_cores",
                        type=int,
                        help="Number of cores to use for parallelization.",
                        default=-1)
    parser.add_argument("--random_state",
                        type=int,
                        help="Random state for data spliting.",
                        default=100)
    return parser.parse_args()


def tokenize_texts(texts, n_cores=1):
    tokens = [[] for _ in texts]
    for i, doc in tqdm(enumerate(nlp.pipe(texts, n_process=n_cores)),
                       total=len(texts),
                       dynamic_ncols=True, leave=False):
        tokens[i] = [x.lemma_.lower().strip() for x in doc]
    return tokens


if __name__ == "__main__":
    args = parse_args()

    # Check files' existence
    assert os.path.isfile(args.raw_data_path), "File for articles not found"

    # Check if output file already exists
    if os.path.isfile(args.output_path):
        warnings.warn("Output file already exists. It will be overwritten.")

    print("Loading the raw MFRC data...")
    mfrc = pd.read_csv(args.raw_data_path)
    print("\tNumber of examples:", len(mfrc))
    print("Finished loading the raw MFRC data...")

    print("Processing the annotations...")
    # Only keep unique examples
    mfrc_unique = pd.DataFrame()
    mfrc_unique["text"] = mfrc.text.unique()

    for i in ["authority", "care", "fairness", "loyalty", "sanctity", "none"]:
        mfrc_unique[i] = 0
    mfrc_unique.set_index("text", inplace=True)

    key_map = {
        "Thin Morality": "none",
        "Non-Moral": "none",
        "Care": "care",
        "Purity": "sanctity",
        "Authority": "authority",
        "Loyalty": "loyalty",
        "Proportionality": "fairness",
        "Equality": "fairness"
    }
    for i, row in tqdm(mfrc.iterrows(), dynamic_ncols=True,
                       total=len(mfrc), leave=False):
        text, fs = row["text"], row["annotation"].split(",")
        for f in fs:
            mfrc_unique.loc[text, key_map[f]] += 1
    print("Finished processing the annotations.")

    # Save
    mfrc_unique["text"] = mfrc_unique.index
    mfrc_unique.index = range(len(mfrc_unique))

    # Splitting into training and test sets
    print("Splitting the data into training and test sets...")

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity", "none"]
    for foundation in foundations:
        if foundation == "none":
            # Seen but not highlighted by anyone
            y = mfrc_unique[["care", "authority", "fairness",
                             "loyalty", "sanctity"]].max(1) == 0
            y = y.astype(int)
            mfrc_unique[f"{foundation}_label"] = 0
            mfrc_unique.loc[y[y == 1].index, f"{foundation}_label"] = 1

            y_train, y_test = train_test_split(mfrc_unique[f"{foundation}_label"],
                                               test_size=0.1,
                                               shuffle=True,
                                               stratify=mfrc_unique[f"{foundation}_label"],
                                               random_state=args.random_state)

            mfrc_unique[f"{foundation}_train"] = 0
            mfrc_unique.loc[y_train.index, f"{foundation}_train"] = 1
        else:
            pos = mfrc_unique[foundation] >= 1
            neg = mfrc_unique[foundation] == 0
            pos, neg = pos.astype(int), neg.astype(int)
            # Label explanation:
            # -1: the sentence has not been seen by anyone assigned this foundation
            #  0: the sentence has been seen at least once with this foundation,
            #     but was not highlighted
            # +1: the sentence has been seen at least once with this foundation,
            #     and was highlighted at least once
            mfrc_unique[f"{foundation}_label"] = -1
            mfrc_unique.loc[pos[pos == 1].index, f"{foundation}_label"] = 1
            mfrc_unique.loc[neg[neg == 1].index, f"{foundation}_label"] = 0
            y = mfrc_unique[(mfrc_unique[f"{foundation}_label"] == 0) |
                            (mfrc_unique[f"{foundation}_label"] == 1)][f"{foundation}_label"]
            y = y.astype(int)
            y_train, y_test = train_test_split(y, test_size=0.1, shuffle=True,
                                               stratify=y,
                                               random_state=args.random_state)
            mfrc_unique[f"{foundation}_train"] = -1
            mfrc_unique.loc[y_train.index, f"{foundation}_train"] = 1
            mfrc_unique.loc[y_test.index, f"{foundation}_train"] = 0

    print("Finished train-test-split.")

    print("Performing k-fold splitting on the training set...")

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity", "none"]
    for f in foundations:
        # Not seen => fold = -1
        mfrc_unique[f + "_fold"] = -1

        # Perform kfold split on the training set
        index = mfrc_unique[mfrc_unique[f + "_label"] >= 0].index
        index = np.array(index)
        labels = mfrc_unique[mfrc_unique[f + "_label"] >= 0][f + "_label"]
        skf = StratifiedKFold(n_splits=10, random_state=args.random_state, shuffle=True)
        a = []
        for fold_no, (train_idx, test_idx) in enumerate(skf.split(index, labels)):
            idx = index[test_idx]
            mfrc_unique.loc[idx, f + "_fold"] = fold_no + 1
            a.extend(idx)
        assert sorted(a) == sorted(index)

        # Test => fold = 0
        index_test = mfrc_unique[mfrc_unique[f + "_train"] == 0].index
        mfrc_unique.loc[index_test, f + "_fold"] = 0

    print("Finished k-fold splitting.")

    print("Tokenizing the texts...")
    mfrc_unique["tokens"] = tokenize_texts(mfrc_unique.text.tolist(), n_cores=args.n_cores)
    print("Finished tokenizing the texts.")
    print("Saving data to CSV file...")
    mfrc_unique.to_csv(args.output_path)
    print("Done.")
