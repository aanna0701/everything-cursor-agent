# Source Code Directory

This directory contains the main Python source code for the project.

## Structure

```
src/
├── __init__.py
├── train.py       # Training script
├── eval.py        # Evaluation script
├── inference.py   # Inference script
├── models/        # Model definitions
├── data/          # Data loading and preprocessing
├── utils/         # Utility functions
└── config.py      # Configuration management
```

## Guidelines

- Follow PEP 8 style guide
- Use type hints on all functions
- Keep functions small (<50 lines)
- One class/concept per file
- Use `pathlib.Path` for file operations
- Use `logging` instead of `print`
