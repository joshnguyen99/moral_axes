import spacy
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import make_concepts_from_lexicon
from gensim.models import KeyedVectors
import argparse
import os
import warnings

emb_path = "data/word2vec_embeddings/glove.twitter.27B.200d.txt"
# no_header=True because the glove vector file doesn't have the
# (number of words, dimension) header default for a typical word2vec file
embedding = KeyedVectors.load_word2vec_format(
    emb_path,
    binary=False,
    no_header=True
)
nlp = spacy.load("en_core_web_md")

# Hand-craft a lexicon
lexicon = {
    "care": ["kindness", "compassion", "nurture", "empathy", "suffer", "cruel", "hurt", "harm"],
    "fairness": ["fairness", "equality", "patriot", "fidelity", "cheat", "fraud", "unfair", "injustice"],
    "loyalty": ["loyal", "team", "patriot", "fidelity", "betray", "treason", "disloyal", "traitor"],
    "authority": ["authority", "obey", "respect", "tradition", "subversion", "disobey", "disrespect", "chaos"],
    "sanctity": ["purity", "sanctity", "sacred", "wholesome", "impurity", "depravity", "degradation", "unnatural"]
}

# Create concept vectors
concepts = make_concepts_from_lexicon(lexicon=lexicon, model=embedding,
                                      verbose=False, normalize=True)


def tokenize(text):
    doc = nlp(text)
    return [tok.text.lower().strip() for tok in doc]


def score_tokens(document, concepts, embedding):
    def score_one_concept(tokens_vectors, concept):
        sims = embedding.cosine_similarities(concept, tokens_vectors)
        return np.mean(sims)

    if type(document) == str:
        tokens = tokenize(document)
    else:
        tokens = document
    tokens_vectors = []
    for w in tokens:
        if w not in embedding:
            continue
        w = embedding[w]
        tokens_vectors.append(w)

    if len(tokens_vectors) <= 0:
        tokens_vectors.append(np.zeros(embedding["x"].shape))

    scores = {}
    for concept, concept_vector in concepts.items():
        scores[concept] = score_one_concept(concept=concept_vector,
                                            tokens_vectors=tokens_vectors)
    scores = pd.Series(scores)
    return scores


def predict_df(df, text_col, output_path):
    df[[f"{f}_score" for f in concepts.keys()]] = 0
    for i, row in tqdm(df.iterrows(), desc="Scored", dynamic_ncols=True,
                       unit=" examples", leave=False, total=len(df)):
        scores = score_tokens(row[text_col], concepts, embedding)
        for concept, score in scores.items():
            df.loc[i, f"{concept}_score"] = score
    df.to_csv(output_path)


def parse_args():
    parser = argparse.ArgumentParser("Score texts using embedding similarity.")
    parser.add_argument("--data",
                        type=str,
                        help="Path to the data file (CSV).",
                        required=True)
    parser.add_argument("--text_col",
                        type=str,
                        help="Name of the column in the data file that contains the texts.",
                        required=True)
    parser.add_argument("--verbose",
                        type=int,
                        help="Whether to print messages (1) or not (0).",
                        default=1)
    parser.add_argument("--output",
                        type=str,
                        help="Path to the output file (CSV).",
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data
    text_col = args.text_col
    verbose = bool(args.verbose)
    output_path = args.output

    # Check if the data file exists
    assert os.path.exists(data_path), f"Data file does not exist at {data_path}."

    # Check if the output file exists
    if os.path.exists(output_path):
        warnings.warn(f"Output file already exists at {output_path}. It will be overwritten.")

    # Load the texts
    if verbose:
        print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Score the texts
    if verbose:
        print(f"Scoring texts...")

    predict_df(df=df,
               text_col=text_col,
               output_path=output_path
               )

    if verbose:
        print("Done!")
