import pandas as pd
import spacy
from tqdm import tqdm
import argparse
import os
import warnings
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

nlp = spacy.load("en_core_web_md", disable=["ner", "senter"])


def parse_args():
    parser = argparse.ArgumentParser(description='Create MFRC data')
    parser.add_argument("--raw_data_path",
                        type=str,
                        help="Path to the MFRC data from Huggingface.",
                        default="data/MFTC_V4_text.json")
    parser.add_argument("--output_path",
                        type=str,
                        help="Path to the output file. In .csv format.",
                        default="data/mftc_preprocessed.csv",
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
    with open(args.raw_data_path, 'r') as f:
        mftc = json.load(f)
    print("Finished loading the raw MFRC data...")

    print("Processing the annotations...")

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity", "none"]

    key_map = {
        "non-moral": "none",
        "nm": "none",
        "nh": "none",
        "care": "care",
        "harm": "care",
        "purity": "sanctity",
        "degradation": "sanctity",
        "authority": "authority",
        "subversion": "authority",
        "loyalty": "loyalty",
        "betrayal": "loyalty",
        "fairness": "fairness",
        "cheating": "fairness"
    }

    # "tweet", "authority", "care", "fairness", "loyalty", "sanctity", "none"
    mftc_df = []
    for corpus in mftc:
        for tweet in corpus["Tweets"]:
            counts = {f: 0 for f in foundations}
            tweet_text = tweet["tweet_text"]
            annots = tweet["annotations"]
            for annot in annots:
                # Get the annotations of this annotator
                labels = annot["annotation"].split(",")
                # Map the annotation to the foundation
                labels = [key_map[l] for l in labels]
                # Increment the counts
                for label in labels:
                    counts[label] += 1
            mftc_df.append([tweet_text] + [counts[label] for label in foundations])
    mftc_df = pd.DataFrame(mftc_df, columns=["tweet"] + foundations)
    print(len(mftc_df))
    print("Finished processing the annotations.")

    # Splitting into training and test sets
    print("Splitting the data into training and test sets...")

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity", "none"]
    for foundation in foundations:
        if foundation == "none":
            # Seen but not highlighted by anyone
            y = mftc_df[["care", "authority", "fairness",
                         "loyalty", "sanctity"]].max(1) == 0
            y = y.astype(int)
            mftc_df[f"{foundation}_label"] = 0
            mftc_df.loc[y[y == 1].index, f"{foundation}_label"] = 1

            y_train, y_test = train_test_split(mftc_df[f"{foundation}_label"],
                                               test_size=0.1,
                                               shuffle=True,
                                               stratify=mftc_df[f"{foundation}_label"],
                                               random_state=args.random_state)

            mftc_df[f"{foundation}_train"] = 0
            mftc_df.loc[y_train.index, f"{foundation}_train"] = 1
        else:
            pos = mftc_df[foundation] >= 1
            neg = mftc_df[foundation] == 0
            pos, neg = pos.astype(int), neg.astype(int)
            # Label explanation:
            # -1: the sentence has not been seen by anyone assigned this foundation
            #  0: the sentence has been seen at least once with this foundation,
            #     but was not highlighted
            # +1: the sentence has been seen at least once with this foundation,
            #     and was highlighted at least once
            mftc_df[f"{foundation}_label"] = -1
            mftc_df.loc[pos[pos == 1].index, f"{foundation}_label"] = 1
            mftc_df.loc[neg[neg == 1].index, f"{foundation}_label"] = 0
            y = mftc_df[(mftc_df[f"{foundation}_label"] == 0) |
                        (mftc_df[f"{foundation}_label"] == 1)][f"{foundation}_label"]
            y = y.astype(int)
            y_train, y_test = train_test_split(y, test_size=0.1, shuffle=True,
                                               stratify=y,
                                               random_state=args.random_state)
            mftc_df[f"{foundation}_train"] = -1
            mftc_df.loc[y_train.index, f"{foundation}_train"] = 1
            mftc_df.loc[y_test.index, f"{foundation}_train"] = 0

    print("Finished train-test-split.")

    print("Performing k-fold splitting on the training set...")

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity", "none"]
    for f in foundations:
        # Not seen => fold = -1
        mftc_df[f + "_fold"] = -1

        # Perform kfold split on the training set
        index = mftc_df[mftc_df[f + "_label"] >= 0].index
        index = np.array(index)
        labels = mftc_df[mftc_df[f + "_label"] >= 0][f + "_label"]
        skf = StratifiedKFold(n_splits=10, random_state=args.random_state, shuffle=True)
        a = []
        for fold_no, (train_idx, test_idx) in enumerate(skf.split(index, labels)):
            idx = index[test_idx]
            mftc_df.loc[idx, f + "_fold"] = fold_no + 1
            a.extend(idx)
        assert sorted(a) == sorted(index)

        # Test => fold = 0
        index_test = mftc_df[mftc_df[f + "_train"] == 0].index
        mftc_df.loc[index_test, f + "_fold"] = 0

    print("Finished k-fold splitting.")

    # Save
    mftc_df.index = range(len(mftc_df))
    print("Tokenizing the texts...")
    mftc_df["tokens"] = tokenize_texts(mftc_df.tweet.tolist(), n_cores=args.n_cores)
    print("Finished tokenizing the texts.")
    print("Saving data to CSV file...")
    mftc_df.to_csv(args.output_path)
    print("Done.")
