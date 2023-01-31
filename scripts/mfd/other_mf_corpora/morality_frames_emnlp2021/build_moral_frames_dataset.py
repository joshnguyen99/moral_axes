"""
Load and process the moral frames dataset.
"""

import pandas as pd
import json

FOUNDATIONS = ["authority", "care", "fairness", "loyalty", "sanctity"]

paths = {
    "authority": "data/combined_annotations_authority_subversion.json",
    "care": "data/combined_annotations_care_harm.json",
    "fairness": "data/combined_annotations_fairness_cheating.json",
    "loyalty": "data/combined_annotations_loyalty_betrayal.json",
    "sanctity": "data/combined_annotations_sanctity_degradation.json"
}


def read_json(foundation):
    with open(paths[foundation], mode="r") as f:
        result = json.load(f)
    return result


data_per_foundation = {f: read_json(f) for f in FOUNDATIONS}

# Columns = ["tweet_id", text", "author_party", "issue", "foundation"]
data_all = []
for foundation, d_foundation in data_per_foundation.items():
    for tweet_id, d in d_foundation.items():
        text = d["text"]
        party = "democratic" if d["author-label"] == 1 else "republican"
        issue = d["issue"]
        row = [tweet_id, text, party, issue, foundation]
        data_all.append(row)

data = pd.DataFrame(
    data_all,
    columns=["tweet_id", "text", "author_party", "issue", "foundation"],
)

print(data)

# Save to csv
data.to_csv("data/moral_frames_dataset.csv", index=True)
