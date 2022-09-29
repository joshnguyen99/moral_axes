# Moral Foundations Dictionaries

## Processing the dictionaries

### Data sources
We look at three verions of the MFD.
- The original version can be downloaded from [here](https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic). It is a `.dic` file containing words and word stems. This file has been processed by Negar Mokhberian *et al.* to create `MFD_original.csv` from [here](https://github.com/negar-mokhberian/Moral_Foundation_FrameAxis/blob/main/moral_foundation_dictionaries/MFD_original.csv).
- The MFD 2.0 is also a `.dic` file and can be downloaded from [here](https://osf.io/whjt2).
- The eMFD is a `.csv` file and can be downloaded from [here](https://osf.io/ufdcz).

### Processing

Ensure that the following 3 files are in the `data/lexicons` folder:
- `MFD_original.csv`
- `mfd2.0.dic`
- `eMFD_wordlist.csv`

Then run
```sh
# Current dir: scripts/mfd
python build_mfd_lexicons.py
```

This will process the MFD and MFD2.0 and save them in the JSON format like
```json
{
    "love": ["care_virtue"],
    "rebellion": ["loyalty_vice", "authority_vice"]
}
```

The eMFD will not be changed. It will look like this
| word    |   care_p |   fairness_p |   loyalty_p |   authority_p |   sanctity_p |   care_sent |   fairness_sent |   loyalty_sent |   authority_sent |   sanctity_sent |
|:--------|---------:|-------------:|------------:|--------------:|-------------:|------------:|----------------:|---------------:|-----------------:|----------------:|
| brought |     0.18 |     0.114286 |        0.08 |     0.0965517 |    0.0533333 |   -0.235404 |       -0.310015 |     -0.0997833 |        -0.402207 |        -0.13255 |

After running the script, you should have 3 files in the `data/lexicons` folder:
- `mfd_original.json`
- `mfd2.json`
- `eMFD_wordlist.csv`


## Scoring texts using the MFD

Assume you have a CSV file in `corpus.csv` where the texts are in the column `text`. To score these texts, run
```sh
# Current dir: scripts/mfd
# Input file
DATA_DIR=corpus.csv
# Name of the column containing the texts
TEXT_COL=text
# Dictionary version (mfd, mfd2 or emfd)
VERSION=mfd2
# Whether to capture the virtue and vice scores separately
SENTIMENT=0
# Whether to normalize the scores
NORMALIZE=1
# Print progress
VERBOSE=1
# Number of cores used in processing the texts
N_JOBS=-1
# Output file
OUTPUT_FILE=corpus_scores_mfd2.csv
# Score
python score_mfd.py --data $DATA_DIR --text_col $TEXT_COL --version $VERSION \
--sentiment $SENTIMENT --normalize $NORMALIZE --verbose $VERBOSE --n_jobs $N_JOBS \
--output $OUTPUT_FILE
```

The output will be a CSV file with each row representing a text, and each column a score for a foundation (e.g., "care_virtue" if sentiment is true, "care" if sentiment is false).