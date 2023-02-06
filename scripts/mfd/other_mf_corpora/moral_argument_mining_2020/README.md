# Moral Argument Mining Dataset

This dataset accompanies the following 2 papers:

*Jonathan Kobbe, Ines Rehbein, Ioana Hulpuș, and Heiner Stuckenschmidt. 2020. Exploring Morality in Argumentation. In Proceedings of the 7th Workshop on Argument Mining, pages 30–40, Online. Association for Computational Linguistics.*

*Henning Wachsmuth, Nona Naderi, Yufang Hou, Yonatan Bilu, Vinodkumar Prabhakaran, Tim Alberdingk Thijm, Graeme Hirst, and Benno Stein. 2017. Computational Argumentation Quality Assessment in Natural Language. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 176–187. Association for Computational Linguistics.*

## Data
1. Download the Dagstuhl-15512-ArgQuality dataset [here](https://zenodo.org/record/3973285) as `argquality_corpus.csv`
2. Download the 2 datasets with Moral Foundations annotations [here](https://github.com/dwslab/Morality-in-Arguments) respectively as `dagstuhl_morality_1.csv` and `dagstuhl_morality_2.csv`.

3. Ensure the `argquality_corpus.csv`, `dagstuhl_morality_1.csv` and `dagstuhl_morality_2.csv` are saved in `scripts/mfd/other_mf_corpora/moral_argument_mining_2020/data`

4. Then run 

```bash
# Current dir: scripts/mfd/other_mf_corpora/moral_argument_mining_2020
python build_moral_args_dataset.py
```

**Explanation for `build_moral_args_dataset.py`**: 
- The original column `MF1`, `MF2` and `MF3` in `dagstuhl_morality_1.csv` and `dagstuhl_morality_2.csv` contains all moral foundations annotated for each argument. We join the 2 datasets on 'argument' column to get all the Moral Foundation labels annotated by 2 annonators for each argument.

- And then join this combined dataset with `argquality_corpus.csv` on '#id' columns to get text content for each argument.

- After executing the above script, The output will be automatically saved to `scripts/mfd/other_mf_corpora/data/moral_argument_mining_2020/data/moral_args_dataset.csv` for evaluation. 

- We will perform inference on the above 2 datasets.

- The final evaluation format is

|    |           argument | care                                                                                                                      | fairness   | loyalty   | authority   |sanctity|
|---:|-------------------:|:--------------------------------------------------------------------------------------------------------------------------|:---------------|:--------|:-------------|:-------------|
|  0 | ‘......’     | 1     | 1     | 0     | 0     | 0 |
