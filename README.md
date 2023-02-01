# Moral Foundations Analyses

## Setting up a Python environment

We will be using Anaconda. First, create an environment.

```bash
$ conda create --name moral_axes python=3.8.5
$ conda activate moral_axes
```

Install packages and dependencies.

```bash
(moral_axes) $ pip install -r requirements_deb.txt
```

## Preparing for NLP packages

### NLTK

For NLTK, download the stopwords.

```bash
$ python -m nltk.downloader stopwords
```

### spaCy

For spaCy, install the `en_core_web_md` pipeline version `3.1.0`. We need this specific version to reproduce the moral foundations news dataset (more in `mfd`).

```bash
$ python -m spacy download en_core_web_md-3.1.0 --direct
```

### GloVe

For GloVe, download the `glove.twitter.27B.200d` embedding.

```
$ cd scripts
$ mkdir data
$ mkdir data/word2vec_embeddings
$ wget https://nlp.stanford.edu/data/glove.twitter.27B.zip -P data/word2vec_embeddings/
$ unzip data/word2vec_embeddings/glove.twitter.27B.zip -d data/word2vec_embeddings/
```

Optionally, remove unused files.

```bash
$ rm data/word2vec_embeddings/glove.twitter.27B.zip
$ rm data/word2vec_embeddings/glove.twitter.27B.25d.txt
$ rm data/word2vec_embeddings/glove.twitter.27B.50d.txt
$ rm data/word2vec_embeddings/glove.twitter.27B.100d.txt
```