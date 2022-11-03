import pandas as pd
import spacy
from tqdm import tqdm
import argparse
import os
import warnings
import json

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

    # Save
    mftc_df.index = range(len(mftc_df))
    print("Tokenizing the texts...")
    mftc_df["tokens"] = tokenize_texts(mftc_df.tweet.tolist(), n_cores=args.n_cores)
    print("Finished tokenizing the texts.")
    print("Saving data to CSV file...")
    mftc_df.to_csv(args.output_path)
    print("Done.")
