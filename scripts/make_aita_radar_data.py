"""
Script for making AITA data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from tqdm import tqdm
import os

FOUNDATIONS = ["authority", "care", "fairness", "loyalty", "sanctity"]

client = MongoClient(host="localhost", port=27017)
db = client.reddit
subs = db.submissions
cmts = db.comments


def find_post(x):
    return subs.find_one({"_id": x})


def find_cmt_by_id(x):
    return cmts.find_one({"_id": x})


# Load posts and their topics
print("Loading AITA posts and their topics...")
aita = pd.read_csv("data/aita/clusters_mfs.csv", index_col="_id")
print("\tAITA total size:", len(aita))

# Create a dataframe for the radar plot
radar_data = pd.DataFrame(index=aita.index)
columns = ["topic_1", "topic_2"]
for col in columns:
    radar_data[col] = aita[col]

# Reconstruct the verdicts
print("Reconstructing the verdicts ")


verdict_ids = []
verdicts = []
scores = []
v_scores = []

for i in tqdm(radar_data.index, total=len(radar_data)):
    # Get the post and verdict
    post = subs.find_one({"_id": i})
    verdict = post["resolved_verdict"]
    verdict_id = post["verdict_id"]
    scores.append(post["score"])

    # Check if the verdict still exists
    cmt = cmts.find_one({"_id": verdict_id})
    if cmt is None or cmt["label"] != verdict:
        # Either the comment was deleted or the verdict doesn't match
        verdict_ids.append(None)
        verdicts.append(None)
        v_scores.append(0)
    else:
        verdict_ids.append(verdict_id)
        verdicts.append(verdict)
        v_scores.append(cmt["score"])

radar_data["verdict"] = verdicts
radar_data["verdict_id"] = verdict_ids
radar_data["score"] = scores
radar_data["verdict_score"] = v_scores
print("\tData size after reconstructing the verdicts:", len(radar_data))

# Remove posts with no verdicts and posts with INFO verdicts
print("Removing posts with no verdicts and posts with INFO verdicts...")
radar_data = radar_data[np.logical_not(radar_data["verdict"].isna())]
radar_data = radar_data[radar_data["verdict"] != "INFO"]
print("\tData size:", len(radar_data))

# Get the MF scores for posts and verdicts
print("Getting the MF scores for posts and verdicts...")
FOUNDATIONS = ["authority", "care", "fairness", "loyalty", "sanctity"]
for f in FOUNDATIONS:
    radar_data[f + "_score"] = 0
for f in FOUNDATIONS:
    radar_data["v_" + f + "_score"] = 0
for i in tqdm(radar_data.index, total=len(radar_data)):
    post = subs.find_one({"_id": i})
    for f in FOUNDATIONS:
        radar_data.loc[i, f + "_score"] = post["mf_scores"][f]

    cmt = cmts.find_one({"_id": radar_data.loc[i, "verdict_id"]})
    for f in FOUNDATIONS:
        radar_data.loc[i, "v_" + f + "_score"] = cmt["mf_scores"][f]

# Add YA and NA columns: (YTA and ESH) -> YA and (NAH and NTA) -> NA
print("Adding YA and NA columns...")
radar_data["YA"] = radar_data["verdict"].isin(["YTA", "ESH"]).astype(int)
radar_data["NA"] = radar_data["verdict"].isin(["NTA", "NAH"]).astype(int)

# Save the data
print("Saving data...")
radar_data.to_csv("data/aita/radar_plot_data/posts_and_verdicts_scored.csv")
print("Done!")
