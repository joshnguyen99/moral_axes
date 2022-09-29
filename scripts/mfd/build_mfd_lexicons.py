"""
This script processes the MFD lexicons from their original formats.
- For MFD and MFD 2.0, the .dic files will be converted to .json files of the
  form {word: [foundation1_sentiment1, foundation2_sentiment2,...]}.
- For eMFD, the csv file won't be changed.

Data sources:
- MFD_original.csv: https://github.com/negar-mokhberian/Moral_Foundation_FrameAxis/blob/main/moral_foundation_dictionaries/MFD_original.csv
- mfd2.0.dic: https://osf.io/whjt2
- eMFD_wordlist.csv: https://osf.io/ufdcz

Ensure that the above files are in the data/lexicons folder.

Note: The original MFD lexicon is a .dic file containing words and word stems. It can be found at
https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic.
This file has been processed by Negar Mokhberian et al. to create MFD_original.csv.
"""

from collections import defaultdict
import json
import pandas as pd
import os

MFD_PATH = os.path.join("data", "lexicons", "MFD_original.csv")
MFD2_PATH = os.path.join("data", "lexicons", "mfd2.0.dic")
EMFD_PATH = os.path.join("data", "lexicons", "eMFD_wordlist.csv")

NEW_MFD_PATH = os.path.join("data", "lexicons", "mfd_original.json")
NEW_MFD2_PATH = os.path.join("data", "lexicons", "mfd2.json")


def process_mfd():
    FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "sanctity"]
    SENTIMENTS = ["virtue", "vice"]
    FOUNDATION_MAP = {
        "authority": "authority",
        "fairness": "fairness",
        "harm": "care",
        "ingroup": "loyalty",
        "purity": "sanctity",
        "general_morality": "general_morality"
    }

    mfd = pd.read_csv(MFD_PATH, index_col="word")
    lexicon = defaultdict(list)
    for word, row in mfd.iterrows():
        f, sent = row["category"].lower().strip(), row["sentiment"].lower().strip()
        f = FOUNDATION_MAP[f]
        # Remove the "general morality" category
        if f not in FOUNDATIONS or sent not in SENTIMENTS:
            continue
        lexicon[word].append(f"{f}_{sent}")

    # Save the lexicon
    with open(NEW_MFD_PATH, "w") as f:
        json.dump(lexicon, f, indent=4)

    print("MFD lexicon saved to ", NEW_MFD_PATH)


def process_mfd2():
    """
    Adapted from https://osf.io/nserp
    """
    nummap = dict()
    mfd2 = defaultdict(list)
    wordmode = True
    with open(MFD2_PATH, 'r') as f:
        for line in f.readlines():
            ent = line.strip().split()
            if line[0] == '%':
                wordmode = not wordmode
            elif len(ent) > 0:
                if wordmode:
                    wordkey = ''.join([e for e in ent if e not in nummap.keys()])
                    mfd2[wordkey].extend([nummap[e] for e in ent if e in nummap.keys()])
                else:
                    nummap[ent[0]] = "_".join(ent[1].split("."))
    with open(NEW_MFD2_PATH, "w") as f:
        json.dump(mfd2, f, indent=4)

    print("MFD 2.0 lexicon saved to ", NEW_MFD2_PATH)


if __name__ == "__main__":
    print("Building MFD lexicons...")
    print("Checking for data files...")
    for path in [MFD_PATH, MFD2_PATH, EMFD_PATH]:
        if os.path.exists(path):
            print(f"\t{path} exists")
        else:
            raise FileNotFoundError(f"{path} does not exist")

    print("Building the MFD original lexicon...")
    process_mfd()
    print("Building the MFD 2.0 lexicon...")
    process_mfd2()
    print("Done!")
