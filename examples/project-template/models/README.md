# Models Directory

This directory stores trained model checkpoints, weights, and artifacts.

## Structure

```
models/
├── checkpoints/  # Training checkpoints (epoch_N.pt)
├── best/         # Best model weights (best.pt)
└── exported/     # Exported models (ONNX, TorchScript, etc.)
```

## Guidelines

- **DO NOT commit model files** to git (they're too large)
- Use `.gitignore` to exclude `.pt`, `.pth`, `.onnx`, `.pkl` files
- Document model architecture and training configs
- Version models using run IDs or timestamps

## Example Usage

```python
from pathlib import Path

models_dir = Path("models/checkpoints")
checkpoint_path = models_dir / "epoch_010.pt"
best_model_path = models_dir / "best.pt"
```
