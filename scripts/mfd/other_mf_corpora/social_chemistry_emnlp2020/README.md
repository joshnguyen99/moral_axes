# Social Chemistry Dataset 101

This dataset is for the paper:
*Social Chemistry 101: Learning to Reason about Social and Moral Norms.
Maxwell Forbes, Jena D. Hwang, Vered Shwartz, Maarten Sap, Yejin Choi
EMNLP 2020*



## Data
1. Download the dataset [here](https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip). Save the `social-chem-101.tsv` file to `data`.

2. Ensure the `social-chem-101.tsv` file is saved in `scripts/mfd/other_mf_corpora/social_chemistry_emnlp2020/data`

3. Then run 

```bash
# Current dir: scripts/mfd/other_mf_corpora/moral_integrity__acl2022
python build_social_chem_dataset.py
```

**Explanation for `build_social_chem_dataset.py`**: 
- The original column `rot-moral-foundations` in `social-chem-101.tsv` contains all moral foundations annotated for each rule-of-thumb (ROT). 

- We perform inference on the test set. The `split` column indicates which examples are part of this set. )

- After executing the above script, The output will be automatically saved to `scripts/mfd/other_mf_corpora/data/social_chemistry_emnlp2020/data/social_chem_dataset.csv`. The format is

|    |           rot | care                                                                                                                      | fairness   | loyalty   | authority   |sanctity|
|---:|-------------------:|:--------------------------------------------------------------------------------------------------------------------------|:---------------|:--------|:-------------|:-------------|
|  0 | ‘......’     | 1     | 1     | 0     | 0     | 0 |

## Description
The description of the dataset can be found [here](https://github.com/mbforbes/social-chemistry-101#dataset-format).