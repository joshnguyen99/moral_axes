from collections import defaultdict
import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import os
import warnings
import re
from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS

nltk_stopwords = stopwords.words('english')

stop_words = set(list(nltk_stopwords) +
                 list(ENGLISH_STOP_WORDS) +
                 list(STOP_WORDS))

nlp = spacy.load("en_core_web_md")


def parse_args():
    parser = argparse.ArgumentParser(description='Create concepts from word embeddings and lexicons')
    parser.add_argument("--articles_path",
                        type=str,
                        help="Path to the file containing the coded news articles. In .pkl format.",
                        default="coded_news.pkl")
    parser.add_argument("--highlights_path",
                        type=str,
                        help="Path to file containing highlights. In .csv format.",
                        default="highlights_raw.csv"
                        )
    parser.add_argument("--output_path",
                        type=str,
                        help="Path to the output file. In .csv format.",
                        default="sentence_mf_counts.csv",
                        )
    parser.add_argument("--n_cores",
                        type=int,
                        help="Number of cores to use for parallelization.",
                        default=1)
    return parser.parse_args()


def filter_hl(hl):
    """
    Ensures that each document has been coded by at least two coders
    that differed in their assigned foundation
    """
    grouped_doc = hl.groupby(hl.document_id).nunique()
    grouped_doc = grouped_doc[grouped_doc.assigned_domain >= 2]
    grouped_doc = grouped_doc[np.logical_and(
        grouped_doc.coder_id <= 15, grouped_doc.coder_id >= 2)]

    keep_docs = grouped_doc.index.values
    hl = hl[hl.document_id.isin(keep_docs)]

    return hl


def normalize(s):
    x = re.sub(r'"', "'", s)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(" \. \. \.", " ...", x)
    x = re.sub("…", "...", x)
    x = re.sub('“', "'", x)
    x = re.sub('”', "'", x)
    x = re.sub('‘', "'", x)
    x = re.sub('’', "'", x)
    x = re.sub('\xad', "", x)
    return x


def overlap(intv1, intv2):
    """
    Check if two integer intevals overlap.
    intv1 = [start1, end1)
    intv2 = [start2, end2)
    """
    s1, e1 = intv1
    s2, e2 = intv2

    # If any of the intervals is empty
    if e1 - s1 <= 0 or e2 - s2 <= 0:
        return False

    if e2 <= e1:
        return e2 > s1

    return e1 > s2


def construct_pos_neg(coded_news, hl, nlp, coder_id, document_id, assigned_domain):
    """
    Construct the positive and negative examples for a given coder, document,
    and assigned domain. A positive example is a sentence that was highlighted
    by the coder (in part or in full).
    Returns:
        pos_texts: list of sentences that were highlighted
        pos_fs: list of foundations corresponding to the positive sentences
        neg_texts: list of sentences that were not highlighted
    """
    # Get the article
    article = coded_news.loc[document_id].text_normalized
    article = nlp(article)

    # Get the highlights that were made by the coder when they were assigned
    # the given domain
    h = hl[(hl.coder_id == coder_id) &
           (hl.document_id == document_id) &
           (hl.assigned_domain == assigned_domain)]

    # Check if no highlights were recorded
    if len(h) == 0:
        return [], [], []

    # Get the highlighted segments and their corresponding foundations
    highlights = h.content_normalized
    foundations = h.assigned_domain

    # Get the intervals of the article where a foundation is recorded
    pos_ints = []
    for hl_text, foundation in zip(highlights, foundations):
        try:
            start = article.text.index(hl_text)
            end = start + len(hl_text)
            pos_ints.append((start, end, foundation))
        except ValueError:
            # The highlighted segment cannot be found in the article
            continue

    # Sort the positive intervals by whichever comes first
    pos_ints = sorted(pos_ints, key=lambda x: x[0])
    pos_founds = [x[2] for x in pos_ints]
    pos_ints = [(x[0], x[1]) for x in pos_ints]

    # Get the sentences that were highlighted (in part or in full)
    pos_texts, pos_fs, neg_texts = [], [], []
    for sent in article.sents:
        intv = (sent.start_char, sent.end_char)
        count = 0
        for intv2, f in zip(pos_ints, pos_founds):
            if intv2[1] > intv[0] and intv2[0] < intv[1] and overlap(intv, intv2):
                # Part of the sentence was highlighted
                pos_texts.append(sent.text)
                pos_fs.append(f)
            else:
                count += 1
        if count >= len(pos_ints):
            # The sentence was not highlighted
            neg_texts.append(sent.text)
    return pos_texts, pos_fs, neg_texts


def preprocess(texts, nlp, progress_bar=False):
    input_type = type(texts)
    if input_type == str:
        texts = [texts]
    tokens = []
    for doc in tqdm(nlp.pipe(texts, n_process=1),
                    desc="Processed", disable=not progress_bar,
                    dynamic_ncols=True, unit=" sentences"):
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

        tokens.append(tokens_df.token.values.tolist())

    if input_type == str:
        return tokens[0]
    return tokens


def parallelize_dataframe(df, func, n_cores=40):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def process_df(df):
    df["tokens"] = preprocess(df.sentence, nlp=nlp, progress_bar=False)
    return df


if __name__ == "__main__":

    args = parse_args()

    # Check files' existence
    assert os.path.isfile(args.articles_path), "File for articles not found"
    assert os.path.isfile(args.highlights_path), "File for highlights not found"

    # Check if output file already exists
    if os.path.isfile(args.output_path):
        warnings.warn("Output file already exists. It will be overwritten.")

    print("Loading the raw highlights...")
    # Load Highlights
    hl = pd.read_csv(args.highlights_path, index_col=0)
    hl = hl[hl.focus_duration >= 45 * 60]
    # Excluded this person as their number of
    # highlights exceeded the average number of
    # highlights-per-coder by 2 standard deviations
    hl = hl[hl['coder_id'] != 549]
    print("\tNumber of highlights: {}".format(len(hl)))
    print("Finished loading the raw highlights.")

    print("Loading the articles...")
    # Load articles
    coded_news = pd.read_pickle(args.articles_path)
    print("\tNumber of articles: {}".format(len(coded_news)))
    print("Finished loading the articles.")

    print("Preprocessing the highlights and articles...")

    # Filter the highlights based on number of assigned foundations
    hl = filter_hl(hl)

    # Normalize the articles and highlights by replacing bad characters
    hl["content_normalized"] = hl.content.astype("unicode").map(normalize)
    coded_news["text_normalized"] = coded_news.text.astype(
        "unicode").map(normalize)

    print("Finished preprocessing the highlights and articles.")

    print("Constructing the positive and negative examples...")

    keys = ["authority", "care", "fairness", "loyalty", "sanctity", "none",
            "authority_seen", "care_seen", "fairness_seen", "loyalty_seen", "sanctity_seen"]
    foundations = ["authority", "care", "fairness", "loyalty", "sanctity"]
    sentence_to_mfs = defaultdict(lambda: {f: 0 for f in keys})
    for coder_id, document_id, assigned_domain in \
            tqdm(hl.groupby(["coder_id", "document_id", "assigned_domain"]).count().index,
                 desc="Processed", dynamic_ncols=True,
                 unit=" coder-document-assignment"):
        pos_texts, pos_fs, neg_texts = construct_pos_neg(coded_news=coded_news,
                                                         hl=hl,
                                                         nlp=nlp,
                                                         coder_id=coder_id,
                                                         document_id=document_id,
                                                         assigned_domain=assigned_domain)
        if len(pos_texts) == 0 and len(neg_texts) == 0:
            continue
        for t, f in zip(pos_texts, pos_fs):
            sentence_to_mfs[t][f + "_seen"] += 1
            sentence_to_mfs[t][f] += 1
        for t in neg_texts:
            sentence_to_mfs[t][f + "_seen"] += 1
            sentence_to_mfs[t]["none"] += 1

    print("Constructing table of sentences and their corresponding foundations...")
    df = pd.DataFrame(0, index=sentence_to_mfs.keys(), columns=foundations)
    for t, mf_counts in tqdm(sentence_to_mfs.items(),
                             desc="Processed",
                             dynamic_ncols=True,
                             unit=" sentences"):
        for f, count in mf_counts.items():
            df.loc[t, f] = count

    df["sentence"] = df.index
    df = df[["sentence", "authority", "care", "fairness", "loyalty", "sanctity", "none",
             "authority_seen", "care_seen", "fairness_seen", "loyalty_seen", "sanctity_seen"]]
    df.loc[:, "authority":] = df.loc[:, "authority":].astype(int)
    df.index = range(len(df))

    print("\tTotal number of sentences: {}".format(len(df)))

    print("Finished constructing positive and negative examples.")

    print("Preprocessing the sentences...")

    df = parallelize_dataframe(df, process_df, n_cores=args.n_cores)
    # Number of tokens
    df["length"] = df.tokens.map(len)

    # Filtering out sentences with fewer than 3 tokens
    df = df[df.length >= 3]
    df.index = range(len(df))

    print("\tNumber of sentences after filtering: {}".format(len(df)))
    print("Finished preprocessing the sentences.")

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
                                               random_state=100)

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
                                               random_state=100)
            df[f"{foundation}_train"] = -1
            df.loc[y_train.index, f"{foundation}_train"] = 1
            df.loc[y_test.index, f"{foundation}_train"] = 0

    print("Finished train-test-split.")

    print("Saving data to CSV file...")

    df.to_csv(args.output_path)

    print("Done.")
