# Data Directory

This directory contains datasets used for training, validation, and testing.

## Structure

```
data/
├── raw/          # Original, unprocessed data
├── processed/    # Cleaned and preprocessed data
└── external/     # External datasets (if any)
```

## Guidelines

- **DO NOT commit large datasets** (>10MB) to git
- Use `.gitignore` to exclude data files
- Document data sources and preprocessing steps in `data/README.md` or project docs
- Keep data paths configurable via config files (not hardcoded)

## Example Usage

```python
from pathlib import Path

data_dir = Path("data/processed")
train_path = data_dir / "train.jsonl"
val_path = data_dir / "val.jsonl"
```
