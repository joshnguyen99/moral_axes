# Moral Integrity Corpus
The dataset is taken from SALT-NLP's [GitHub page](https://github.com/SALT-NLP/mic.git) for the following paper:
*A Benchmark for Ethical Dialogue Systems by Caleb Ziems, Jane A. Yu, Yi-Chia Wang, Alon Y. Halevy, Diyi Yang*


## Data
1. Download the dataset [here](https://www.dropbox.com/sh/m46z42nce8x0ttk/AABuSZiA6ESyrJNWmgTPrfuRa?dl=0). Save the `MIC.csv` file to `data`.

2. Ensure the `MIC.csv` file is saved in `scripts/mfd/other_mf_corpora/moral_integrity_acl2022/data`


3. Then run 

```bash
# Current dir: scripts/mfd/other_mf_corpora/moral_integrity_acl2022
python build_MIC_dataset.py
```

**Explanation for `build_MIC_dataset.py`**: 
- The original column `moral` in `MIC.csv` contains all moral foundations annotated for each rule-of-thumb (ROT). We ignore `liberty`.

- We will perform inference on the test set. The `split` column indicates which examples are part of this set. 

- After executing the script, The output will be automatically saved to `scripts/mfd/other_mf_corpora/moral_integrity_acl2022/data/MIC_dataset.csv`. 

- The final evaluation format is

|    |           rot | care                                                                                                                      | fairness   | loyalty   | authority   |sanctity|
|---:|-------------------:|:--------------------------------------------------------------------------------------------------------------------------|:---------------|:--------|:-------------|:-------------|
|  0 | ‘......’     | 0     | 0     | 1     | 0     | 1 |


## Description
The below is taken from SALT-NLP's [GitHub page](https://github.com/SALT-NLP/mic.git) for the following paper:

**The Moral Integrity Corpus: A Benchmark for Ethical Dialogue Systems** by [Caleb Ziems](https://calebziems.com/), Jane A. Yu, [Yi-Chia Wang](https://scholar.google.com/citations?user=9gMgFPQAAAAJ&hl=en), [Alon Y. Halevy](https://scholar.google.com/citations?user=F_MI0pcAAAAJ&hl=en), [Diyi Yang](https://www.cc.gatech.edu/~dyang888/)


## *What is MIC?* 

Open-domain or "chit-chat" conversational agents often reflect insensitive, hurtful, or contradictory viewpoints that erode a user’s trust in the integrity of the system. Moral integrity is one important pillar for building trust. 

`MIC` is a dataset that can help us understand chatbot behaviors through their latent values and moral statements. `MIC` contains 114k annotations, with 99k distinct "Rules of Thumb" (RoTs) that capture the moral assumptions of 38k chatbot replies to open-ended prompts. These RoTs represent diverse moral viewpoints, with the following distribution of underlying moral foundations: 

* ![51%](https://progress-bar.dev/51) **Care:** wanting someone or something to be safe, healthy, and happy. (58k chatbot replies)
* ![21%](https://progress-bar.dev/21) **Fairness:** wanting to see individuals or groups treated equally or equitably. (24k)
* ![19%](https://progress-bar.dev/19) **Liberty:** wanting people to be free to make their own decisions. (22k)
* ![19%](https://progress-bar.dev/19) **Loyalty:** wanting unity and seeing people keep promises or obligations to an in-group. (22k)
* ![18%](https://progress-bar.dev/18) **Authority:** wanting to respect social roles, duties, privacy, peace, and order. (20k)
* ![11%](https://progress-bar.dev/11) **Sanctity:** wanting people and things to be clean, pure, innocent, and holy. (13k)

