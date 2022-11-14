from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, RobertaModel, AutoModelForSequenceClassification, AutoConfig
import torch
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def load_model(path):
    state_dict = torch.load(path)["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
    model = AutoModelForSequenceClassification.from_pretrained(state_dict=new_state_dict,
                                                               pretrained_model_name_or_path=model_name)
    return model


def predict(X, model, device="cuda:0", batch_size=64):
    model.to(device)
    model.eval()
    y_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), leave=False):
            x = X[i:i+batch_size]
            inputs = tokenizer(x, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)
            outputs = torch.softmax(outputs.logits, dim=1)
            outputs = outputs[:, 1]
            y_scores.extend(outputs.detach().cpu().numpy())
    return y_scores


labelers = {
    "authority": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/authority/checkpoints/trial-1-20221109-075214-762383/last.ckpt",
    "care": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/care/checkpoints/trial-1-20221114-034608-371367/last.ckpt",
    "fairness": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/fairness/checkpoints/trial-1-20221109-081812-371459/last.ckpt",
    "loyalty": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/loyalty/checkpoints/trial-1-20221109-072631-151450/last.ckpt",
    "sanctity": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/sanctity/checkpoints/trial-1-20221114-034611-278774/last.ckpt"
}


def predict_df(df, text_col, output_path):
    for f, path in labelers.items():
        print(f)
        print(path)
        model = load_model(path)
        y_score = predict(X=df[text_col].tolist(), model=model, batch_size=16)
        df[f"{f}_score"] = y_score
    df.to_csv(output_path)


print("cm_ambig")
cm_ambig = pd.read_csv("mfd/data/ethics_commonsense_scoring_results/dataset/cm_ambig.csv", header=None)
cm_ambig.rename(columns={0: "input"}, inplace=True)
predict_df(cm_ambig, text_col="input",
           output_path="mfd/data/ethics_commonsense_scoring_results/predictions/cm_ambig.csv")

print("cm_test_hard")
cm_test_hard = pd.read_csv("mfd/data/ethics_commonsense_scoring_results/dataset/cm_test_hard.csv")
predict_df(cm_test_hard, text_col="input",
           output_path="mfd/data/ethics_commonsense_scoring_results/predictions/cm_test_hard.csv")

print("cm_test")
cm_test = pd.read_csv("mfd/data/ethics_commonsense_scoring_results/dataset/cm_test.csv")
predict_df(cm_test, text_col="input",
           output_path="mfd/data/ethics_commonsense_scoring_results/predictions/cm_test.csv")

print("cm_train")
cm_train = pd.read_csv("mfd/data/ethics_commonsense_scoring_results/dataset/cm_train.csv")
predict_df(cm_train, text_col="input",
           output_path="mfd/data/ethics_commonsense_scoring_results/predictions/cm_train.csv")
