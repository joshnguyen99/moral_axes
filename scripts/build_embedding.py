import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import warnings
from scipy.sparse import save_npz
import json


def preprocess_texts(texts, progress_bar=False):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from nltk.corpus import stopwords
    from spacy.lang.en.stop_words import STOP_WORDS
    import spacy

    nltk_stopwords = stopwords.words('english')
    stop_words = set(list(nltk_stopwords) +
                     list(ENGLISH_STOP_WORDS) +
                     list(STOP_WORDS))
    nlp = spacy.load("en_core_web_md")
    input_type = type(texts)
    if input_type == str:
        texts = [texts]
    tokens = [[] for _ in range(len(texts))]
    for i, doc in tqdm(enumerate(nlp.pipe(texts, n_process=-1)),
                       desc="Processed", disable=not progress_bar,
                       dynamic_ncols=True, unit=" examples",
                       leave=False, total=len(texts)):

        # Lemmatize, lowercase
        toks = [tok.lemma_.lower().strip() for tok in doc]

        # Remove stopwords
        toks = [tok for tok in toks if tok not in stop_words]

        # Only keep alphabetic tokens
        toks = [tok for tok in toks if tok.isalpha()]

        # Only keep tokens with at least 3 characters
        toks = [tok for tok in toks if len(tok) >= 3]

        tokens[i] = toks

    if input_type == str:
        return tokens[0]
    return tokens


def create_bow_embedding(texts, progress_bar=False):
    from sklearn.feature_extraction.text import CountVectorizer
    tokens = preprocess_texts(texts, progress_bar)
    bow_vec = CountVectorizer(tokenizer=lambda tokens: tokens,
                              lowercase=False, stop_words=None,
                              min_df=5,
                              max_df=0.99
                              )
    embedding = bow_vec.fit_transform(tokens)
    return embedding, bow_vec


def create_spacy_embedding(texts, progress_bar=False):
    import spacy
    nlp = spacy.load("en_core_web_md")
    input_type = type(texts)
    if input_type == str:
        texts = [texts]
    embedding = np.zeros((len(texts), 300))
    for i, doc in tqdm(enumerate(nlp.pipe(texts, n_process=-1)),
                       desc="Processed", disable=not progress_bar,
                       dynamic_ncols=True, unit=" examples",
                       leave=False, total=len(texts)):
        embedding[i, :] = doc.vector
    return embedding


def create_glove_embedding(texts, progress_bar=False):
    import spacy
    from utils import make_one_concept
    from gensim.models import KeyedVectors
    nlp = spacy.load("en_core_web_md")
    input_type = type(texts)
    if input_type == str:
        texts = [texts]
    glove_filename = "data/word2vec_embeddings/glove.twitter.27B.200d"
    word2vec_output_file = glove_filename + '.word2vec'
    glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    X_glove = np.zeros((len(texts), 200))
    for i, doc in tqdm(enumerate(nlp.pipe(texts, n_process=-1)),
                       desc="Processed", disable=not progress_bar,
                       dynamic_ncols=True, unit=" examples",
                       leave=False, total=len(texts)):
        tokens = [tok.text.lower().strip() for tok in doc]
        X_glove[i] = make_one_concept(model=glove, word_list=tokens,
                                      normalize=True)
    return X_glove


def create_sentence_roberta_embedding(texts, progress_bar=False):
    from sentence_transformers import SentenceTransformer
    import torch
    input_type = type(texts)
    if input_type == str:
        texts = [texts]

    model = SentenceTransformer("stsb-roberta-large")

    if torch.cuda.is_available():
        model.to("cuda")

    X_bert = model.encode(texts, convert_to_numpy=True,
                          show_progress_bar=progress_bar)
    return X_bert


def parse_args():
    parser = argparse.ArgumentParser(description='Embed texts for downstream classification')
    parser.add_argument("--type",
                        type=str,
                        default="bow",
                        help="Embedding type (bow, spacy, glove or sentence_roberta).",
                        required=True,
                        choices=["bow", "spacy", "glove", "sentence_roberta"])
    parser.add_argument("--data",
                        type=str,
                        default="mfd/data/mf_corpora_merged.csv",
                        help="Path to the data file (CSV).",
                        required=True)
    parser.add_argument("--text_col",
                        type=str,
                        default="sentence",
                        help="Name of the column in the data file that contains the texts.",
                        required=True)
    parser.add_argument("--verbose",
                        type=int,
                        help="Whether to print messages (1) or not (0).",
                        default=1)
    parser.add_argument("--output",
                        type=str,
                        help="Path to output file.",
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Check input file
    assert os.path.isfile(args.data), "Data file not found"

    # Load data
    data = pd.read_csv(args.data)

    # Check if text column exists
    assert args.text_col in data.columns, f"Column {args.text_col} does not exist in the dataset."
    texts = data[args.text_col].tolist()

    # Check if the output file exists
    if os.path.exists(args.output):
        warnings.warn(f"Output file already exists at {args.output}. It will be overwritten.")

    if args.type == "bow":
        if args.verbose:
            print("Creating BoW embedding...")
        embedding, bow_vec = create_bow_embedding(texts, progress_bar=args.verbose)
        print(type(bow_vec.vocabulary_))
        save_npz(args.output, embedding)
        if args.verbose:
            print("Saved BoW embedding to", args.output)
        vectorizer_output = os.path.join(*os.path.split(args.output)[:-1], "bow_vectorizer.json")
        print(f"Also saving the vocabulary to {vectorizer_output}")
        with open(vectorizer_output, "w") as f:
            json.dump(bow_vec.vocabulary_, f, indent=4)

    elif args.type == "spacy":
        if args.verbose:
            print("Creating Spacy embedding...")
        embedding = create_spacy_embedding(texts, progress_bar=args.verbose)
        np.save(args.output, embedding)
        if args.verbose:
            print("Saved Spacy embedding to", args.output)

    elif args.type == "glove":
        if args.verbose:
            print("Creating GloVe embedding...")
        embedding = create_glove_embedding(texts, progress_bar=args.verbose)
        np.save(args.output, embedding)
        if args.verbose:
            print("Saved GloVe embedding to", args.output)

    elif args.type == "sentence_roberta":
        if args.verbose:
            print("Creating Sentence RoBERTa embedding...")
        embedding = create_sentence_roberta_embedding(texts, progress_bar=args.verbose)
        np.save(args.output, embedding)
        if args.verbose:
            print("Saved Sentence RoBERTa embedding to", args.output)
