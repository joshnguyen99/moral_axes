# Moral Foundations Dictionaries

This folder contains scripts for building the moral foundations (MF) lexicons and datasets to train MF labelers.

## Creating MF Lexicons

### Data sources
We look at three verions of the MFD.
- The original version can be downloaded from [here](https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic). It is a `.dic` file containing words and word stems. This file has been processed by Negar Mokhberian *et al.* to create `MFD_original.csv` from [here](https://github.com/negar-mokhberian/Moral_Foundation_FrameAxis/blob/main/moral_foundation_dictionaries/MFD_original.csv).
- The MFD 2.0 is also a `.dic` file and can be downloaded from [here](https://osf.io/whjt2).
- The eMFD is a `.csv` file and can be downloaded from [here](https://osf.io/ufdcz).

### Processing

Create a data folder
```sh
# Current dir: scripts/mfd
mkdir data
mkdir data/lexicons
```

Ensure that the following 3 files are in `data/lexicons`:
- `MFD_original.csv`
- `mfd2.0.dic`
- `eMFD_wordlist.csv`

Then run
```sh
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


## Constructing Datasets for Moral Foundations Labeling

We will use three sources:
- Annotated news articles. This is used to construct the eMFD lexicon above. Download and save the following two files to `data`:
  - `coded_news.pkl` from [here](https://osf.io/5r96b). It contains the news articles.
  - `highlights_raw.csv` from [here](https://osf.io/52qfe). It contains the highlights for each article, the foundation each highlight is about, and who annotated each highlight.
- Annotated Tweets. The ID and annotations for each tweet is in the `MFTC_V4.json` file from [here](https://osf.io/k5n7y/). You can use the IDs to retrieve the original tweets' text. Some of them might have been deleted in the past, so we contacted the authors for the original tweets. Save the `MFTC_V4_text.json` file in `data`.
- Annotated Reddit comments. Download the `final_mfrc_data.csv` file from [here](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC/resolve/main/final_mfrc_data.csv) and save it to `data`.

You should have the following files in `data`:
- `coded_news.pkl`
- `highlights_raw.csv`
- `MFTC_V4.json`
- `final_mfrc_data.csv`

Now we will process each dataset separately, and then combine them into a single dataset.

### Processing the news dataset

Each news article contains a number of highlights for a foundation. Each highlight can be any subsequence. We will create a dataset mapping a sentence to a moral foundation. For example, if sentence S has label `care`, it means S (entirely or in part) was highlighted as containing *care*.

The following script will create such a dataset.

```sh
python construct_emfd_data.py --output_path data/sentence_mf_counts.csv
```

The output file `data/sentence_mf_counts.csv` contains the following columns:
- `authority`,...,`sanctity`: The number of times a sentence was labeled with this foundation.
- `authority_seen`,...,`sanctity_seen`: How many times a sentence was assigned to be labeled with this foundation. For example, if a sentence has `authority_seen = 1`, it was assigned to be labeled with *authority* once.
- `none`: The number of times a sentence was not labeled with any foundation.

Then, build a one-vs-all dataset for each foundation.

```sh
python prepare_one_vs_all_data.py --output_path data/sentence_mf_one_v_all.csv
```

### Processing the Twitter dataset

For this dataset, we ignore the sentiment (virtue/vice) component of a foundation. For example, *purity* and *degradation* will be mapped to *sanctity*.

```sh
python construct_mftc_data.py --output_path data/mftc_preprocessed.csv
```

### Processing the Reddit dataset

For this dataset, we map both "proportionality" and "equality" to *fairness*. Also, the author-defined label "thin morality" will be mapped to "none".

```sh
python construct_mfrc_data.py --output_path data/mfrc_preprocessed.csv
```

### Merging into one dataset

Ensure that after processing the three datasets, you have the following files in `data`:
- `data/sentence_mf_one_v_all.csv`
- `data/mfrc_preprocessed.csv`
- `data/mftc_preprocessed.csv`

This simple script will merge all these dataset into one and save the result to `data/mf_corpora_merged.csv`

```sh
python merge_mf_corpora.py
```

The result in `data/mf_corpora_merged.csv` contains the following columns:
- `sentence`: the raw text. This will be used an inputs for training the labelers/classifiers.
- `tokens`: the tokens in the sentence. This can be used for tallying words signifying a foundation, or for creating a bag-of-words representation.
- `authority_label`,...,`none_label`: the label for each foundation. These will be used as the binary output in training the classifiers. The label is `1` if the sentence contains the foundation, and `0` when it doesn't. Some sentences contain a value of `-1`, which means the they were not seen during labeling. In this case, the sentences should be ignored in training.
- `authority_train`,...,`none_train`: a binary indicator for whether the sentence was used in training to recognize each foundation. For example, if `authority_train` is `1`, it means the sentence was used in the training set for the *authority*-vs-all classifier. Again, some sentences contain a value of `-1`, which means they should be ignored.
- `authority_fold`,...,`none_fold`: the fold number for k-fold cross-validation. The folds for training are from `1` to `10`. Fold `0` indicates the test set. Fold `-1` indicates the sentence should be ignored in training, for the same reason as above.

### Example usage

If you want to train a classifier to decide whether the foundation *authority* exists in text, here's how to load the data.

```python
import pandas as pd
foundation = "authority"

# Load all sentences
dataset = pd.read_csv('data/mf_corpora_merged.csv', index_col=0)

# Remove sentences with label = -1
dataset = dataset[dataset[f'{foundation}_label'] != -1]

# Training set
training_set = dataset[dataset[f'{foundation}_fold'] != 0]
X_train, y_train = training_set['sentence'], training_set[f'{foundation}_label']
# Use this for 10-fold CV
train_folds = training_set[f'{foundation}_fold']

# Test set
test_set = dataset[dataset[f'{foundation}_fold'] == 0]
X_test, y_test = test_set['sentence'], test_set[f'{foundation}_label']
```