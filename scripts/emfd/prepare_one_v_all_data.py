"""
Load the sentence MFD dataset.
For each foundation, perform k-fold cross validation.
"""

from signal import default_int_handler
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse
import os
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description='Create one-vs-all data for the sentence MFD dataset.')
    parser.add_argument("--input_path",
                        type=str,
                        help="Path to the moral foundation counts file. In .csv format.",
                        default="data/sentence_mf_counts.csv")
    parser.add_argument("--output_path",
                        type=str,
                        help="Path to the output file. In .csv format.",
                        default="data/sentence_mf_one_v_all.csv",
                        )
    parser.add_argument("--random_state",
                        type=int,
                        help="Random state for data spliting.",
                        default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check file's existence
    assert os.path.isfile(args.input_path), "Input CSV file not found"

    # Check if output file already exists
    if os.path.isfile(args.output_path):
        warnings.warn("Output file already exists. It will be overwritten.")

    # Read the sentence MF counts file
    df = pd.read_csv(args.input_path, index_col=0)

    print("Splitting the data into training and test sets...")

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity", "none"]
    for foundation in foundations:
        if foundation == "none":
            # Seen but not highlighted by anyone
            y = df[["care", "authority", "fairness",
                    "loyalty", "sanctity"]].max(1) == 0
            y = y.astype(int)
            df[f"{foundation}_label"] = 0
            df.loc[y[y == 1].index, f"{foundation}_label"] = 1

            y_train, y_test = train_test_split(df[f"{foundation}_label"],
                                               test_size=0.1,
                                               shuffle=True,
                                               stratify=df[f"{foundation}_label"],
                                               random_state=args.random_state)

            df[f"{foundation}_train"] = 0
            df.loc[y_train.index, f"{foundation}_train"] = 1
        else:
            pos = df[foundation] >= 1
            neg = (df[foundation + "_seen"] >= 1) & (df[foundation] == 0)
            pos, neg = pos.astype(int), neg.astype(int)
            # Label explanation:
            # -1: the sentence has not been seen by anyone assigned this foundation
            #  0: the sentence has been seen at least once with this foundation,
            #     but was not highlighted
            # +1: the sentence has been seen at least once with this foundation,
            #     and was highlighted at least once
            df[f"{foundation}_label"] = -1
            df.loc[pos[pos == 1].index, f"{foundation}_label"] = 1
            df.loc[neg[neg == 1].index, f"{foundation}_label"] = 0
            y = df[(df[f"{foundation}_label"] == 0) |
                   (df[f"{foundation}_label"] == 1)][f"{foundation}_label"]
            y = y.astype(int)
            y_train, y_test = train_test_split(y, test_size=0.1, shuffle=True,
                                               stratify=y,
                                               random_state=args.random_state)
            df[f"{foundation}_train"] = -1
            df.loc[y_train.index, f"{foundation}_train"] = 1
            df.loc[y_test.index, f"{foundation}_train"] = 0

    print("Finished train-test-split.")

    print("Performing k-fold splitting on the training set...")

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity", "none"]
    for f in foundations:
        # Not seen => fold = -1
        df[f + "_fold"] = -1

        # Perform kfold split on the training set
        index = df[df[f + "_label"] >= 0].index
        index = np.array(index)
        labels = df[df[f + "_label"] >= 0][f + "_label"]
        skf = StratifiedKFold(n_splits=10, random_state=args.random_state, shuffle=True)
        a = []
        for fold_no, (train_idx, test_idx) in enumerate(skf.split(index, labels)):
            idx = index[test_idx]
            df.loc[idx, f + "_fold"] = fold_no + 1
            a.extend(idx)
        assert sorted(a) == sorted(index)
        
         # Test => fold = 0
        index_test = df[df[f + "_train"] == 0].index
        df.loc[index_test, f + "_fold"] = 0
        
        

    print("Finished k-fold splitting.")

    print("Removing unnecessary columns...")

    df = df.drop(["authority", "care", "fairness", "loyalty", "sanctity", "none",
                  "authority_seen", "care_seen", "fairness_seen", "loyalty_seen", "sanctity_seen",
                  "length"],
                 axis=1)

    print("Finished removing unnecessary columns.")
    
    print("Saving data to CSV file...")
    
    df.to_csv(args.output_path)
    
    print("Done.")
    