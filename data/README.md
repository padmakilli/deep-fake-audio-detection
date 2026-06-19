# Dataset

This project trains on the **Fake-or-Real (FoR)** dataset and (optionally) cross-validates
on **ASVspoof 2019 LA**.

- Fake-or-Real: https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
- ASVspoof 2019: https://www.asvspoof.org/index2019.html

## Download (Kaggle CLI)

```bash
pip install kaggle                       # add your token to ~/.kaggle/kaggle.json
kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset -p data/raw
unzip -q data/raw/the-fake-or-real-dataset.zip -d data/raw
```

`python scripts/download_data.py` prints the same commands.

## Expected layout

Point `data.root` in `config/config.yaml` at the **for-norm** directory. Each split
contains `real/` (genuine human) and `fake/` (deepfake) sub-folders:

```
data/raw/for-norm/
├── training/   { real/  fake/ }
├── validation/ { real/  fake/ }
└── testing/    { real/  fake/ }
```

Labels are assigned automatically: `real -> 1 (genuine)`, `fake -> 0 (deepfake)`.
Raw audio is git-ignored; only folder placeholders are committed.
