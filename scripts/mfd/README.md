# Moral Foundations Dictionaries

## Processing the dictionaries

### Data sources
We look at three verions of the MFD.
- The original version can be downloaded from [here](https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic). It is a `.dic` file containing words and word stems. This file has been processed by Negar Mokhberian *et al.* to create `MFD_original.csv` from [here](https://github.com/negar-mokhberian/Moral_Foundation_FrameAxis/blob/main/moral_foundation_dictionaries/MFD_original.csv).
- The MFD 2.0 is also a `.dic` file and can be downloaded from [here](https://osf.io/whjt2).
- The eMFD is a `.csv` file and can be downloaded from [here](https://osf.io/ufdcz).

### Processing

Ensure that the following 3 files are in the `data/lexicons` folder:
- `MFD_original.csv`
- `mfd2.0.dic`
- `eMFD_wordlist.csv`

Then run
```bash
# Current dir: scripts/mfd
python build_mfd_lexicons.py
```

This will process the MFD and MFD2.0 and save them in the JSON format like
```json
{
    "love": ["care_virtue"],
    "rebellion": ["loyalty_vice", "authority_vice"]
}
```

The eMFD will not be changed. It will look like this
| word    |   care_p |   fairness_p |   loyalty_p |   authority_p |   sanctity_p |   care_sent |   fairness_sent |   loyalty_sent |   authority_sent |   sanctity_sent |
|:--------|---------:|-------------:|------------:|--------------:|-------------:|------------:|----------------:|---------------:|-----------------:|----------------:|
| brought |     0.18 |     0.114286 |        0.08 |     0.0965517 |    0.0533333 |   -0.235404 |       -0.310015 |     -0.0997833 |        -0.402207 |        -0.13255 |

After running the script, you should have 3 files in the `data/lexicons` folder:
- `mfd_original.json`
- `mfd2.json`
- `eMFD_wordlist.csv`