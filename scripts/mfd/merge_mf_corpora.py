import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_paths = {
    "mfnc": "data/sentence_mf_one_v_all.csv",
    "mfrc": "data/mfrc_preprocessed.csv",
    "mftc": "data/mftc_preprocessed.csv",
    "merged": "data/mf_copora_merged.csv"
}


def load_data(name):
    return pd.read_csv(data_paths[name], index_col=0)


mfnc = load_data(name="mfnc")
mfrc = load_data(name="mfrc").rename(columns={"text": "sentence"})
mftc = load_data(name="mftc").rename(columns={"tweet": "sentence"})

mfrc = mfrc.drop(["authority", "care", "fairness", "loyalty", "sanctity", "none"], axis=1)
mftc = mftc.drop(["authority", "care", "fairness", "loyalty", "sanctity", "none"], axis=1)

merged = pd.concat([mfnc, mfrc, mftc], axis=0)
merged.index = range(len(merged))

merged.to_csv("data/mf_corpora_merged.csv")
