# Text Analysis with Moral Foundations

## Scoring Moral Foundations in Texts

### Requirements

First we need to build the MF dictionaries (MFD, MFD2, eMFD), the GloVe lexicons and the models for MF labelers. `cd` to `mfd` and follow the instructions.

For all scroring scripts below, we assume you have a CSV file in `corpus.csv` where the texts are in the column `text`. Also, assume you want to save the output to `corpus_mf_annotated.csv`. Create some environment variables for these

```sh
# Input file
DATA_DIR=corpus.csv
# Name of the column containing the texts
TEXT_COL=text
# Output file
OUTPUT_FILE=corpus_mf_annotated.csv
```

### Scoring using moral foundations dictionaries

```sh
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
# Score
python score_mf_mfd.py --data $DATA_DIR --text_col $TEXT_COL --version $VERSION \
--sentiment $SENTIMENT --normalize $NORMALIZE --verbose $VERBOSE --n_jobs $N_JOBS \
--output $OUTPUT_FILE
```

The output will be a CSV file with each row representing a text, and each column a score for a foundation (e.g., `care_virtue` if sentiment is true, `care` if sentiment is false).

### Scoring using embedding similarity

Check the `score_mf_ddr.py` file. It contains the seed words to construct the axis for each moral foundations. You can change these words as you see fit as well.

```sh
# Score
python score_mf_ddr.py --data $DATA_DIR --text_col $TEXT_COL --verbose $VERBOSE \
--output $OUTPUT_FILE
```

### Scoring using fine-tuned RoBERTa models

Save the checkpoint for each model. Open the `score_mf_roberta.py` file and modify the path to each checkpoint in (i.e., change the path in the `labelers` dictionary).

```sh
# Which device to use (cpu or cuda)
DEVICE=cuda
# Batch size for RoBERTa
BATCH_SIZE=32
# Score
python score_mf_roberta.py --data $DATA_DIR --text_col $TEXT_COL --verbose $VERBOSE \
--device $DEVICE --batch_size $BATCH_SIZE --output $OUTPUT_FILE
```