import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import scipy
import os

if not os.path.exists("data/embeddings_for_classifiers/tfidf_transformers"):
    os.makedirs("data/embeddings_for_classifiers/tfidf_transformers")

BOW_EMB_PATH = "data/embeddings_for_classifiers/bow.npz"

emb_path = BOW_EMB_PATH

X = scipy.sparse.load_npz(emb_path)

data = pd.read_csv("mfd/data/mf_corpora_merged.csv", index_col=0)
foundations = ["authority", "care", "fairness", "loyalty", "sanctity"]
tfidf_transformers = []

for f in foundations:
    idx = data[data[f"{f}_fold"] > 0].index
    X_train = X[idx]
    tfidf = TfidfTransformer().fit(X_train)
    with open(f"data/embeddings_for_classifiers/tfidf_transformers/{f}.pkl", "wb") as f:
        pickle.dump(tfidf, f)
