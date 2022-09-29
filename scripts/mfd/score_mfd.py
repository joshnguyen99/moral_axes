import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import json
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
import argparse
import warnings

nltk_stopwords = stopwords.words('english')
stop_words = set(list(nltk_stopwords) +
                 list(ENGLISH_STOP_WORDS) +
                 list(STOP_WORDS))
nlp = spacy.load("en_core_web_md")

MFD_PATH = os.path.join("data", "lexicons", "mfd_original.json")
assert os.path.exists(MFD_PATH), f"{MFD_PATH} does not exist"
with open(MFD_PATH, "r") as f:
    MFD = json.load(f)

MFD2_PATH = os.path.join("data", "lexicons", "mfd2.json")
assert os.path.exists(MFD2_PATH), f"{MFD2_PATH} does not exist"
with open(MFD2_PATH, "r") as f:
    MFD2 = json.load(f)

EMFD_PATH = os.path.join("data", "lexicons", "eMFD_wordlist.csv")
assert os.path.exists(EMFD_PATH), f"{EMFD_PATH} does not exist"
with open(EMFD_PATH, "r") as f:
    EMFD = pd.read_csv(EMFD_PATH, index_col="word")


def score_mfd(texts,
              sentiment=False,
              normalize=True,
              version=1,
              verbose=True,
              n_jobs=-1):
    """
    Score a text using the Moral Foundations Dictionary.

    args:
        texts (str or list of str): Texts (tweet, sentence, etc.) to be scored.
        sentiment (bool, optional): If true, the text will be scored in the
            virtue and vice for each foundation, with a total of 10 scores
            (e.g., care_virtue, care_vice). Otherwise, the virtue and vice will
            be combined into a single score (e.g., care). Defaults to False.
        normalize (bool, optional): If true, the count for each dimension will
            be divided by the number of tokens in `text`, so that is each score
            is between 0 and 1. Defaults to True.
        version (int, optional): Version of the MFD (1 or 2) to use. 
            Defaults to 1.
        verbose (bool, optional): If true, the progress bar will be shown.
            Defaults to True.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.

    returns:
        scores_df (pd.DataFrame): Dataframe where row = text, column = score. The
            order of the rows is the same as the order of the texts in the inputs.
    """
    if type(texts) == str:
        texts = [texts]

    # Load the dictionary
    assert version in [1, 2], "MFD version must be 1 or 2"
    lexicon = MFD if version == 1 else MFD2

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity"]
    dimensions = ["authority_virtue", "authority_vice",
                  "care_virtue", "care_vice",
                  "fairness_virtue", "fairness_vice",
                  "loyalty_virtue", "loyalty_vice",
                  "sanctity_virtue", "sanctity_vice"]

    # Score the documents
    scores_df = pd.DataFrame(0, columns=dimensions,
                             index=range(len(texts)))
    for i, doc in tqdm(enumerate(nlp.pipe(texts, n_process=n_jobs)),
                       desc="Scored",
                       disable=not verbose,
                       dynamic_ncols=True,
                       unit=" docs",
                       total=len(texts)):
        tokens = [token.text.lower() for token in doc]
        lemmas = [token.lemma_.lower() for token in doc]
        n_tokens = len(tokens)
        for token, lemma in zip(tokens, lemmas):
            cats = []
            # First check if the lexicon contains the token
            if token in lexicon:
                cats = lexicon[token]
            # If not, check if the lemma is in the lexicon
            elif lemma in lexicon:
                cats = lexicon[lemma]
            # Increment the count for each category
            scores_df.loc[i, cats] += 1
        # Normalize the counts, avoiding division by zero
        if normalize and n_tokens > 0:
            scores_df.loc[i] /= n_tokens

    # Merge the virtue and vice scores
    if not sentiment:
        scores_df_merged = pd.DataFrame(0, columns=foundations,
                                        index=range(len(scores_df)))
        for f in foundations:
            scores_df_merged[f] = scores_df[[f + "_virtue", f + "_vice"]].sum(axis=1)
        scores_df = scores_df_merged

    return scores_df


def score_emfd(texts,
               sentiment=False,
               normalize=True,
               verbose=True,
               n_jobs=-1):
    """
    Score a text using the extende Moral Foundations Dictionary.

    args:
        texts (str or list of str): Texts (tweet, sentence, etc.) to be scored.
        sentiment (bool, optional): If true, the text will be scored in the
            virtue and vice for each foundation, with a total of 10 scores
            (e.g., care_virtue, care_vice). Otherwise, the virtue and vice will
            be combined into a single score (e.g., care). Defaults to False.
        normalize (bool, optional): If true, the count for each dimension will
            be divided by the number of tokens in `text`, so that is each score
            is between 0 and 1. The default in the eMFD paper is True.
        verbose (bool, optional): If true, the progress bar will be shown.
            Defaults to True.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.

    returns:
        scores_df (pd.DataFrame): Dataframe where row = text, column = score. The
            order of the rows is the same as the order of the texts in the inputs.
    """
    if type(texts) == str:
        texts = [texts]

    # Load the dictionary
    lexicon = EMFD

    foundations = ["authority", "care", "fairness", "loyalty", "sanctity"]
    dimensions = ["authority_virtue", "authority_vice",
                  "care_virtue", "care_vice",
                  "fairness_virtue", "fairness_vice",
                  "loyalty_virtue", "loyalty_vice",
                  "sanctity_virtue", "sanctity_vice"]

    # Score the documents
    scores_df = pd.DataFrame(0,
                             columns=dimensions if sentiment else foundations,
                             index=range(len(texts)))
    for i, doc in tqdm(enumerate(nlp.pipe(texts, n_process=n_jobs)),
                       desc="Scored",
                       disable=not verbose,
                       dynamic_ncols=True,
                       unit=" docs",
                       total=len(texts)):
        # Preprocess the document like how the eMFD was created
        tokens_df = [[tok.text, tok.ent_type_, tok.tag_] for tok in doc]
        tokens_df = pd.DataFrame(
            tokens_df, columns=["token", "entity", "pos"])

        # Remove entities
        tokens_df = tokens_df[tokens_df.entity == ""]

        # Filter POS tags
        keep_pos = ['NN', 'NNS', 'JJ', 'VB', 'VBD',
                    'VBG', 'VBN', 'VBP', 'VBZ', 'RB']
        tokens_df = tokens_df[tokens_df.pos.isin(keep_pos)]

        # Remove bad characters
        no_chars = ["…", "'ve", "'s", "'ll", "'d", "\"", "'m", "'s", "'re", "–-", '–-', '‘', '’d',
                    '’ll', '’m', '’re', '’s', '’ve', '“', ',,', ',', '(', ')', '.', '”', '\n\n',
                    "@realDonaldTrump", "n't", '\xad']
        tokens_df = tokens_df[np.logical_not(
            tokens_df.token.map(str.lower).isin(no_chars))]

        # Remove stop words
        tokens_df = tokens_df[np.logical_not(
            tokens_df.token.isin(list(stop_words)))]

        # Remove non-alphabetic characters from each token
        tokens_df["token"] = tokens_df.token.apply(
            lambda x: "".join(filter(lambda c: c.isalpha(), x)))

        # Keep tokens with at least 3 characters
        tokens_df = tokens_df[tokens_df.token.apply(len) >= 3]

        # Lowercase tokens
        tokens_df["token"] = tokens_df.token.apply(str.lower)

        tokens = tokens_df.token.values.tolist()

        # Used for normalization
        dimension_counts = pd.Series(0, index=dimensions if sentiment else foundations)

        for token in tokens:
            # Check if the lexicon contains the token
            if token in lexicon.index:
                row = lexicon.loc[token]
            else:
                continue
            scores = row[[f + "_p" for f in foundations]].to_numpy()
            sentiments = row[[f + "_sent" for f in foundations]]
            score_sent_cols = [f + ("_virtue" if s >= 0 else "_vice")
                               for f, s in zip(foundations, sentiments)]

            # Add scores to this text
            if sentiment:
                scores_df.loc[i, score_sent_cols] += scores
                dimension_counts[score_sent_cols] += 1
            else:
                scores_df.loc[i, foundations] += scores
                dimension_counts[foundations] += 1

        # Normalize the scores by counts
        if normalize:
            dimension_counts[dimension_counts == 0] = 1
            scores_df.loc[i, dimension_counts.index] /= dimension_counts

    return scores_df


def parse_args():
    parser = argparse.ArgumentParser("Score texts using the Moral Foundations Dictionary.")
    parser.add_argument("--data",
                        type=str,
                        help="Path to the data file (CSV).",
                        required=True)
    parser.add_argument("--text_col",
                        type=str,
                        help="Name of the column in the data file that contains the texts.",
                        required=True)
    parser.add_argument("--version",
                        type=str,
                        help="The MFD version.",
                        default="mfd",
                        choices=["mfd", "mfd2", "emfd"])
    parser.add_argument("--normalize",
                        type=int,
                        help="Whether to normalize the scores for each text. Normalization differs"
                        " between MFD (2.0) and eMFD.",
                        default=1)
    parser.add_argument("--sentiment",
                        type=int,
                        help="Whether to score the sentiment (virtue and vice) of each foundation "
                             "(1) or not (0).",
                        default=0)
    parser.add_argument("--verbose",
                        type=int,
                        help="Whether to print messages (1) or not (0).",
                        default=1)
    parser.add_argument("--n_jobs",
                        type=int,
                        help="Number of jobs to run in parallel when tokenizing texts.",
                        default=-1)
    parser.add_argument("--output",
                        type=str,
                        help="Path to the output file (CSV).",
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data
    text_col = args.text_col
    version = args.version
    normalize = bool(args.normalize)
    sentiment = bool(args.sentiment)
    verbose = bool(args.verbose)
    output_path = args.output
    n_jobs = args.n_jobs

    # Check if the data file exists
    assert os.path.exists(data_path), f"Data file does not exist at {data_path}."

    # Check if the output file exists
    if os.path.exists(output_path):
        warnings.warn(f"Output file already exists at {output_path}. It will be overwritten.")

    # Load the texts
    if verbose:
        print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, usecols=[text_col])
    texts = df[text_col].to_numpy()
    del df

    # Score the texts
    if verbose:
        print(f"Scoring texts...")
    if version == "mfd":
        scores_df = score_mfd(texts,
                              normalize=normalize,
                              sentiment=sentiment,
                              verbose=verbose,
                              version=1,
                              n_jobs=n_jobs)
    elif version == "mfd2":
        scores_df = score_mfd(texts,
                              normalize=normalize,
                              sentiment=sentiment,
                              verbose=verbose,
                              version=2,
                              n_jobs=n_jobs)
    elif version == "emfd":
        scores_df = score_emfd(texts,
                               normalize=normalize,
                               sentiment=sentiment,
                               verbose=verbose,
                               n_jobs=n_jobs)

    if verbose:
        print(f"Saving scores to {output_path}...")
    scores_df.to_csv(output_path)

    if verbose:
        print("Done!")
