import spacy
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import make_concepts_from_lexicon
from gensim.models import KeyedVectors
from utils import make_one_concept
emb_path = "data/word2vec_embeddings/glove.twitter.27B.200d.word2vec"
embedding = KeyedVectors.load_word2vec_format(emb_path, binary=False)
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


def predict_df(df, text_col):
    df[[f"{f}_score" for f in concepts.keys()]] = 0
    for i, row in tqdm(df.iterrows(), desc="Scored", dynamic_ncols=True,
                       unit=" examples", leave=False, total=len(df)):
        scores = score_tokens(row[text_col], concepts, embedding)
        for concept, score in scores.items():
            df.loc[i, f"{concept}_score"] = score
    return df
