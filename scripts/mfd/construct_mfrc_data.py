import pandas as pd
import spacy
from tqdm import tqdm
import argparse
import os
import warnings

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
    return parser.parse_args()


def tokenize_texts(texts, n_cores=1):
    tokens = [[] for _ in texts]
    for doc in tqdm(nlp.pipe(texts, n_process=n_cores), total=len(texts),
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
    print("Tokenizing the texts...")
    mfrc_unique["tokens"] = tokenize_texts(mfrc_unique.text.tolist(), n_cores=args.n_cores)
    print("Finished tokenizing the texts.")
    print("Saving data to CSV file...")
    mfrc_unique.to_csv(args.output_path)
    print("Done.")
