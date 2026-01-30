# Configs Directory

This directory contains configuration files for experiments, training, and inference.

## Structure

```
configs/
├── train/        # Training configurations
├── eval/         # Evaluation configurations
└── inference/    # Inference/deployment configs
```

## Guidelines

- Use YAML or TOML for config files (not JSON)
- Keep configs version-controlled (they're small)
- Use dataclasses or Pydantic for config validation
- Document all config parameters

## Example Usage

```python
from pathlib import Path
import yaml

config_path = Path("configs/train/default.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)
```
