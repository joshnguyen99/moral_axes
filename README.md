# Measuring Moral Dimensions on Social Media with Mformer

This repository accompanies the following paper:

Tuan Dung Nguyen, Ziyu Chen, Nicholas George Carroll, Alasdair Tran, Colin Klein, and Lexing Xie. **“Measuring Moral Dimensions in Social Media with Mformer”**. *Proceedings of the International AAAI Conference on Web and Social Media* 18 (2024). 

arXiv: https://doi.org/10.48550/arXiv.2311.10219.

## Check out the demo of our Mformer models via this 🤗 [Hugging Face space](https://huggingface.co/spaces/joshnguyen/mformer)!

## Loading Mformer locally

The 5 Mformer models are available on Hugging Face.

| Moral foundation    | Model URL |
| -------- | ------- |
| **Authority**  | https://huggingface.co/joshnguyen/mformer-authority    |
| **Care** | https://huggingface.co/joshnguyen/mformer-care     |
| **Fairness**    | https://huggingface.co/joshnguyen/mformer-fairness    |
| **Loyalty**    | https://huggingface.co/joshnguyen/mformer-loyalty    |
| **Sanctity**    | https://huggingface.co/joshnguyen/mformer-sanctity    |

Here's how to load Mformer. Note that each model's weights are in FP32 format, which totals about 500M per model. If your computer's memory does not accommodate this, you might want to load it in FP16 or BF16 format.

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# Change the foundation name if need be 
FOUNDATION = "authority"
MODEL_NAME = f"joshnguyen/mformer-{FOUNDATION}"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)
```

To perform inference:

```python
# Perform inference
instances = [
    "Earlier Monday evening, Pahlavi addressed a private audience and urged 'civil disobedience by means of non-violence.'",
    "I am a proponent of civil disobedience and logic driven protest only; not non/ irrational violence, pillage & mayhem!"
]

# Encode the instances
inputs = tokenizer(
    instances,
    padding=True,
    truncation=True,
    return_tensors='pt'
).to(model.device)

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# Calculate class probability
probs = torch.softmax(outputs.logits, dim=1)
probs = probs[:, 1]
probs = probs.detach().cpu().numpy()

# Print results
print(f"Probability of foundation {FOUNDATION}:", "\n")
for instance, prob in zip(instances, probs):
    print(instance, "->", prob, "\n")
```

which will print out the following

```bash
Probability of foundation authority:

Earlier Monday evening, Pahlavi addressed a private audience and urged 'civil disobedience by means of non-violence.' -> 0.9462048

I am a proponent of civil disobedience and logic driven protest only; not non/ irrational violence, pillage & mayhem! -> 0.97276026
```

<!-- ###  Setting up a Python environment

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
``` -->