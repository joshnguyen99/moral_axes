import pandas as pd
import os
import warnings
import argparse
from utils.roberta_utils import (
    load_model,  # Load weights from a RobertaForSequenceClassification model
    predict,  # Use a RobertaForSequenceClassification model to predict a list of texts
)

labelers = {
    "authority": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/authority/checkpoints/trial-1-20221109-075214-762383/last.ckpt",
    "care": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/care/checkpoints/trial-1-20221114-034608-371367/last.ckpt",
    "fairness": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/fairness/checkpoints/trial-1-20221109-081812-371459/last.ckpt",
    "loyalty": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/loyalty/checkpoints/trial-1-20221109-072631-151450/last.ckpt",
    "sanctity": "/localdata/u7221462/dev/moral-predictor/experiments/merged/one-vs-all-combined/sanctity/checkpoints/trial-1-20221114-034611-278774/last.ckpt"
}


def predict_df(df, text_col, output_path, device="cuda", batch_size=32):
    for f, path in labelers.items():
        print(f)
        print(path)
        model = load_model(path)
        model.to(device)
        y_score = predict(X=df[text_col].tolist(), model=model, batch_size=batch_size)
        df[f"{f}_score"] = y_score
    df.to_csv(output_path)


def parse_args():
    parser = argparse.ArgumentParser("Score texts using fine-tuned RoBERTa models.")
    parser.add_argument("--data",
                        type=str,
                        help="Path to the data file (CSV).",
                        required=True)
    parser.add_argument("--text_col",
                        type=str,
                        help="Name of the column in the data file that contains the texts.",
                        required=True)
    parser.add_argument("--verbose",
                        type=int,
                        help="Whether to print messages (1) or not (0).",
                        default=1)
    parser.add_argument("--output",
                        type=str,
                        help="Path to the output file (CSV).",
                        required=True)
    parser.add_argument("--device",
                        type=str,
                        help="Device for RoBERTa.",
                        default="cuda")
    parser.add_argument("--batch_size",
                        type=int,
                        help="Batch size.",
                        default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data
    text_col = args.text_col
    verbose = bool(args.verbose)
    output_path = args.output
    device = args.device
    batch_size = args.batch_size

    # Check if the data file exists
    assert os.path.exists(data_path), f"Data file does not exist at {data_path}."

    # Check if the output file exists
    if os.path.exists(output_path):
        warnings.warn(f"Output file already exists at {output_path}. It will be overwritten.")

    # Load the texts
    if verbose:
        print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Score the texts
    if verbose:
        print(f"Scoring texts...")

    predict_df(df=df,
               text_col=text_col,
               output_path=output_path,
               device=device,
               batch_size=batch_size)

    if verbose:
        print("Done!")
