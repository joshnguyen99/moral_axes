# Text Analysis with Moral Foundations

## Scoring Moral Foundations in Texts

We use several methods of scoring moral foundations in text:
1. Word count based on previouly released lexicons (a.k.a. moral foundations dictionaries). More details can be found in the `mfd`'s README.
   - MFD: original word-to-foundation lexicon.
   - MFD 2.0: expert-crafted seed words, then extended using similarity in word2vec space.
   - eMFD: words discovered from human annotation of moral foundations.
2. Embedding similarity. Here we use the 200-dimensional GloVe word embedding trained on a Twitter corpus. 
   - Each foundation (or "concept") is described by a list of keywords. For example, `"loyalty" = ["loyal", "team", "patriot", "fidelity", "betray", "treason"]`.
   - The vector describing the concept *loyalty* is the average of all embeddings for these keywords.
   - A document is transformed into a vector by tokenizing and averaging the embedding of these tokens. This is called the document embedding.
   - To assess the relevance of a document with a concept, we consider the cosine similarity between their embeddings.
3. Logistic regression. This is a multi-label dataset (i.e., a document can be about more than 1 moral foundation, or none), we set this up as one-vs-all classification task. In other words, we trained a model for each foundation.
   - Document embeddings:
     - Sparse: bag-of-words, tfidf
     - Dense: we take the average of all of its tokens' embeddings. Several token embedding methods are used: GloVe, spaCy, Sentence-RoBERTa.
   - We tune hyperparameter for $\ell_2$ regularization using 10-fold cross validation.
4. Transformers. We also set this up as a one-vs-all classification task.
    - We finetune RoBERTa (`roberta-base` from Huggingface) for a binary classification task using the log loss.
    - The AdamW algorithm is used for optimization, along with cosine
    - To avoid overfitting, we control the learning rate and number of training epochs.
    - The final result is 5 binary classifier, one for each moral foundation.
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