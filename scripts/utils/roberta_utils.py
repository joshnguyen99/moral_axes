from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def load_model(path):
    """
    Load weights from a RobertaForSequenceClassification model.

    Args:
        path: path to .ckpt checkpoint
    """
    state_dict = torch.load(path)["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
    model = AutoModelForSequenceClassification\
        .from_pretrained(state_dict=new_state_dict,
                         pretrained_model_name_or_path=model_name)
    return model


def predict(X, model, device="cuda:0", batch_size=64):
    """
    Use a RobertaForSequenceClassification model to predict a list of texts.

    Args:
        X: a list of array of texts
        model: an instance of RobertaForSequenceClassification, with 2 classes
        device: torch device. Defaults to "cuda:0".
        batch_size: batch size. Defaults to 64.

    Returns:
        a list of scores for the texts in X
    """
    model.to(device)
    model.eval()
    y_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), total=len(range(0, len(X), batch_size)),
                      leave=False):
            x = X[i:i + batch_size]
            inputs = tokenizer(x, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)
            outputs = torch.softmax(outputs.logits, dim=1)
            outputs = outputs[:, 1]
            y_scores.extend(outputs.detach().cpu().numpy())
    print("LEN", len(y_scores))
    return y_scores


def token_strip(token):
    cont_char = "Ġ"
    starts_with_cont = token.startswith(cont_char)
    stripped_token = token
    if starts_with_cont:
        stripped_token = token[len(cont_char):]
    is_new_token = starts_with_cont
    return stripped_token, is_new_token


def tokens_to_string_with_attn(tokens, scores,
                               strip_left_special=True,
                               strip_right_special=True,
                               strip_fn=token_strip):
    """
    Assemble a list of tokens into a string, handling sub-word tokens.

    Args:
        tokens: list of RoBERTa tokens
        scores: attention scores for each token
        strip_left_special: Whether to remove the [CLS] token. Defaults to True.
        strip_right_special: Whether to remove the [SEP] token. Defaults to True.
        strip_fn: _description_. Defaults to token_strip.

    Returns:
        (token_list, score_list)
    """
    if strip_left_special:
        tokens = tokens[1:]
        scores = scores[1:]
    if strip_right_special:
        tokens = tokens[:-1]
        scores = scores[:-1]
    token_list = []
    score_list = []
    curr_token = None
    curr_scores = []
    for token, score in zip(tokens, scores):
        token, is_new_token = strip_fn(token)
        if not is_new_token:
            if curr_token is None:
                curr_token = ""
            curr_token += token
            curr_scores.append(score)
        else:
            if curr_token is not None:
                token_list.append(curr_token)
                # Take the average of the attention scores
                # for all sub-word tokens of the word
                score_list.append(np.mean(curr_scores))
            curr_token = token
            curr_scores = [score]
    if len(curr_scores) > 0:
        token_list.append(curr_token)
        score_list.append(np.mean(curr_scores))
    return token_list, score_list


def get_attention_scores(text, models, device="cuda"):
    """
    Get attention scores for text.

    Args:
        text: a string
        models: a dictionary containing models of the type
                {
                    "category1": RobertaForSequenceClassification,
                    "category2": RobertaForSequenceClassification,
                    ...
                }
        device: torch device. Defaults to "cuda"

    Returns:
        A pandas DataFrame with the attention scores for each token. The
        columns are the categories and the rows are the tokens.
    """

    # Tokenize input
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)

    # Get raw tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    tokens_and_attns = None

    all_categories = list(models.keys())

    # Use the models to encode the text
    for foundation in models.keys():
        # Get model
        model = models[foundation]
        model.to(device)

        # Encode the input
        outputs = model(**inputs, output_attentions=True)

        # Extract the attention scores
        # Last RoBERTa layer
        # Shape = (batch_size=1, num_heads=12, seq_len, seq_len)
        # Average over all attention heads
        # First [0]: batch size is 1
        # Second [0]: attention scores for the first token
        attns = outputs.attentions[-1].mean(dim=1)[0][0].cpu().detach().numpy()

        # Merge some sub-word tokens into one word token
        token_list, score_list = \
            tokens_to_string_with_attn(tokens=tokens,
                                       scores=attns,
                                       strip_left_special=False,  # Remove [CLS]
                                       strip_right_special=False,  # Remove [SEP]
                                       )
        if tokens_and_attns is None:
            tokens_and_attns = pd.DataFrame(
                0,
                columns=all_categories,
                index=token_list
            )

        tokens_and_attns[foundation] = score_list

    return tokens_and_attns
