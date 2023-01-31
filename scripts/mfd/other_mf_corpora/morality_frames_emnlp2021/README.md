# Morality Frames dataset

The below is taken from Shamik Roy's [GitHub page](https://github.com/ShamikRoy/Moral-Role-Prediction) for the following paper:

## Data

Ensure the following 5 files are in `scripts/mfd/other_mf_corpora/morality_frames_emnlp2021/data`:
- `combined_annotations_authority_subversion.json`
- `combined_annotations_care_harm.json`
- `combined_annotations_fairness_cheating.json`
- `combined_annotations_loyalty_betrayal.json`
- `combined_annotations_sanctity_degradation.json`

Then run 

```bash
# Current dir: scripts/mfd/other_mf_corpora/morality_frames_emnlp2021
python build_moral_frames_dataset.py
```

The output is saved to `scripts/mfd/other_mf_corpora/data/moral_frames_dataset.csv`. The format is

|    |           tweet_id | text                                                                                                                      | author_party   | issue   | foundation   |
|---:|-------------------:|:--------------------------------------------------------------------------------------------------------------------------|:---------------|:--------|:-------------|
|  0 | Tweet ID | Tweet text | republican/democratic     | aca/immig/...    | authority/...    |


## Description
The below is taken from Shamik Roy's [GitHub page](https://github.com/ShamikRoy/Moral-Role-Prediction) for the following paper:

*Identifying Morality Frames in Political Tweets using Relational Learning; Shamik Roy, Maria Leonor Pacheco and Dan Goldwasser Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021).*

Annotated tweets from 5 moral foundations are contained in 5 separate json files. The annotation labels in the json file correspond to the answers to the following questions. 

care/harm:
    q1. Which entity needs CARE, or is being HARMED? [target of care/harm]
    q2. Which entity is causing the HARM? [entity causing harm]
    q3. Which entity is offering/providing the CARE? [entity providing care]

Fairness/Cheating:
    q1. Fairness or cheating on what? [additional question]
    q2. Fairness for whom or who is being cheated? [target of fairness/cheating]
    q3. Who or What is ensuring fairness or in charge of ensuring fairness? [entity ensuring fairness]
    q4. Who or What is cheating or violating the fairness? [entity doing cheating]
    
Loyalty/Betrayal:
    q1. What are the phrases invoking LOYALTY? [additional question]
    q2. What are the phrases invoking BETRAYAL? [additional question]
    q3. LOYALTY or BETRAYAL to whom or what? [target of loyalty/betrayal]
    q4. Who or what is expressing LOYALTY? [entity being loyal]
    q5. Who or what is doing BETRAYAL? [entity doing betrayal]

Authority/Subversion:
    q1. LEADERSHIP or AUTHORITY on what issue or activity? [additional question]
    q2. Which LEADERSHIP or AUTHORITY is obeyed/praised/justified? [justified authority]
    q3. If the LEADERSHIP or AUTHORITY is obeyed/praised/justified, then praised/obeyed by whom or justified over whom? [justified authority over]
    q4. Which LEADERSHIP or AUTHORITY is disobeyed or failing or criticized? [failing authority]
    q5. If the LEADERSHIP or AUTHORITY is disobeyed or failing or criticized, then failing to lead whom or disobeyed/criticized by whom? [failing authority over]

Purity/Degradation:
    q1. What or who is SACRED, or subject to degradation? [target of purity/degradation]
    q2. Who is ensuring or preserving the sanctity? [entity preserving purity]
    q3. Who is violating the sanctity or who is doing degradation or who is the target of disgust? [entity causing degradation]

Author-Label Mapping:

0 - Republican
1 - Democrat

Topic Mapping:

aca - affordable care act
immig - immigration
abort - abortion
guns - guns
isis - Terrorism 
lgbt - lgbtq




