---
name: coding-standards
description: Python ML/DL coding standards, best practices, and patterns for deep learning model training, inference, and deployment.
---

# Python ML/DL Coding Standards

Coding standards for Python machine learning and deep learning projects.

> **Note**: This skill focuses on ML/DL-specific patterns. For general Python coding standards (type hints, pathlib, logging, immutability), see project rules in `.cursor/rules/python-coding-style.md` or the rules in this library.

## ML/DL Code Quality Principles

### 1. Reproducibility First
- Always set seeds for reproducibility
- Log all hyperparameters and configs
- Version control data and models
- Document environment dependencies

### 2. Research-to-Production Mindset
- Write modular, testable code from day one
- Separate concerns: data, model, training, evaluation
- Use structured configs (dataclasses, Pydantic)
- Plan for deployment even during experimentation

### 3. Device-Agnostic Design
- Never hardcode device (CUDA/MPS/CPU)
- Support multiple compute backends
- Graceful fallback to CPU when GPU unavailable

### 4. Safe Serialization
- Always use `weights_only=True` for untrusted checkpoints
- Prefer safetensors for model weights
- Never use `pickle.load` on untrusted data

## ML/DL Best Practices

### Reproducibility

```python
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms (may impact performance).

    Note:
        Full determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 env var
        and may not be achievable for all operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# Call at start of training
set_seed(config.seed, deterministic=config.deterministic)
```

### Device Management

```python
import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Get compute device.

    Args:
        device_str: "auto", "cuda", "mps", or "cpu"

    Returns:
        torch.device for computation
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# GOOD: Explicit device placement
device = get_device()
model = model.to(device)
batch = {k: v.to(device) for k, v in batch.items()}


# BAD: Hardcoded device
model = model.cuda()  # Fails on non-CUDA systems
```

### Checkpointing

```python
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    epoch: int,
    config: TrainingConfig,
    metrics: dict[str, float],
    path: Path,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        config: Training configuration.
        metrics: Current metrics (loss, accuracy, etc.).
        path: Output path for checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        device: Device to load tensors to.

    Returns:
        Checkpoint dict with epoch, config, metrics.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
```

### Safe Serialization (CRITICAL)

```python
# GOOD: Safe loading with weights_only=True
checkpoint = torch.load(path, weights_only=True)

# GOOD: Use safetensors for model weights
from safetensors.torch import save_file, load_file

def save_model_safe(model: torch.nn.Module, path: Path) -> None:
    save_file(model.state_dict(), path)

def load_model_safe(model: torch.nn.Module, path: Path) -> None:
    state_dict = load_file(path)
    model.load_state_dict(state_dict)


# BAD: Unsafe pickle loading (arbitrary code execution risk)
checkpoint = torch.load(untrusted_path)  # DANGEROUS for untrusted files

# BAD: pickle.load on untrusted data
import pickle
data = pickle.load(untrusted_file)  # NEVER do this with untrusted input
```

## ML/DL Configuration Patterns

### Training Configuration with Dataclasses

```python
# GOOD: ML/DL-specific config with dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration for ML/DL models."""

    # Model
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    num_layers: int = 12

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 10
    gradient_accumulation_steps: int = 1

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)


# BAD: Dictionary config (no validation, typo-prone)
config = {
    "model_name": "bert-base-uncased",
    "batch_size": 32,
    # No validation, no type checking, easy to typo keys
}
```

### ML/DL-Specific Error Handling

```python
# GOOD: ML/DL-specific exceptions
from pathlib import Path


class CheckpointNotFoundError(Exception):
    """Raised when checkpoint file is not found."""


class InvalidCheckpointError(Exception):
    """Raised when checkpoint is corrupted or incompatible."""


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise CheckpointNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, weights_only=True)
    except Exception as e:
        raise InvalidCheckpointError(f"Failed to load checkpoint {path}: {e}") from e

    required_keys = {"model_state_dict", "optimizer_state_dict", "epoch"}
    if not required_keys.issubset(checkpoint.keys()):
        missing = required_keys - checkpoint.keys()
        raise InvalidCheckpointError(f"Checkpoint missing keys: {missing}")

    return checkpoint


# BAD: Generic exception handling
def load_checkpoint(path):
    try:
        return torch.load(path)
    except:  # noqa: E722 - bare except
        return None  # Silently fails
```

## Code Organization

### ML/DL Project Structure

```
project/
├── pyproject.toml          # Project metadata, dependencies (uv/pip)
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── config.py           # Configuration dataclasses
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py      # Dataset classes
│       │   ├── transforms.py   # Data transforms/augmentation
│       │   └── collate.py      # Collate functions
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py         # Base model class
│       │   ├── encoder.py      # Model components
│       │   └── losses.py       # Loss functions
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py      # Training loop
│       │   ├── callbacks.py    # Training callbacks
│       │   └── metrics.py      # Evaluation metrics
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── predictor.py    # Inference wrapper
│       │   └── server.py       # FastAPI serving
│       └── utils/
│           ├── __init__.py
│           ├── logging.py      # Logging setup
│           └── seed.py         # Reproducibility utils
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
├── scripts/
│   ├── train.py                # Training entrypoint
│   ├── evaluate.py             # Evaluation script
│   └── serve.py                # Serving script
├── configs/
│   └── base.yaml               # Default config (optional)
└── outputs/                    # Gitignored, for artifacts
    ├── checkpoints/
    ├── logs/
    └── experiments/
```

### File Naming

```
src/project/
├── config.py              # snake_case for modules
├── training_loop.py       # Descriptive names
├── bert_encoder.py        # Model-specific files

tests/
├── test_dataset.py        # test_ prefix for pytest
├── test_training.py
└── conftest.py            # Pytest fixtures
```

### Import Organization

```python
# GOOD: Organized imports (ruff enforces this)

# Standard library
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local
from project.config import TrainingConfig
from project.models import Encoder


# BAD: Messy imports
from project.config import *  # Star imports
import torch, numpy  # Multiple imports on one line
```

> **Note**: For general import organization rules, see project rules in `.cursor/rules/python-coding-style.md` or the rules in this library.

## ML/DL Documentation Patterns

### Docstring Style (Google Format)

```python
def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> dict[str, float]:
    """Train model for specified epochs.

    Implements standard training loop with validation, checkpointing,
    and early stopping based on validation loss.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Training configuration.

    Returns:
        Dictionary with final metrics:
            - train_loss: Final training loss
            - val_loss: Final validation loss
            - best_epoch: Epoch with best validation loss

    Raises:
        ValueError: If model is not on correct device.
        RuntimeError: If CUDA runs out of memory.

    Example:
        >>> config = TrainingConfig(max_epochs=10)
        >>> metrics = train_model(model, train_loader, val_loader, config)
        >>> print(f"Best epoch: {metrics['best_epoch']}")
    """
    ...
```

### ML/DL-Specific Comments

```python
# GOOD: Explain WHY, not WHAT - especially for ML/DL patterns

# Use gradient accumulation to simulate larger batch sizes
# without increasing GPU memory requirements
for i, batch in enumerate(dataloader):
    loss = model(batch) / config.gradient_accumulation_steps
    loss.backward()

    if (i + 1) % config.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Mixed precision disabled for this layer due to numerical instability
# in softmax computation with FP16 (see issue #123)
with torch.cuda.amp.autocast(enabled=False):
    attention_scores = torch.softmax(scores.float(), dim=-1)


# BAD: State the obvious
# Loop through batches
for batch in dataloader:
    # Calculate loss
    loss = model(batch)
```

## Testing Standards

### Test Structure

```python
import pytest
import torch

from project.models import Encoder


class TestEncoder:
    """Tests for Encoder model."""

    @pytest.fixture
    def encoder(self) -> Encoder:
        """Create encoder instance for testing."""
        return Encoder(input_dim=768, hidden_dim=256)

    def test_forward_shape(self, encoder: Encoder) -> None:
        """Forward pass returns correct output shape."""
        x = torch.randn(4, 10, 768)  # (batch, seq, features)
        output = encoder(x)
        assert output.shape == (4, 10, 256)

    def test_forward_deterministic(self, encoder: Encoder) -> None:
        """Forward pass is deterministic in eval mode."""
        encoder.eval()
        x = torch.randn(2, 5, 768)
        out1 = encoder(x)
        out2 = encoder(x)
        torch.testing.assert_close(out1, out2)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_variable_batch_size(
        self, encoder: Encoder, batch_size: int
    ) -> None:
        """Model handles variable batch sizes."""
        x = torch.randn(batch_size, 10, 768)
        output = encoder(x)
        assert output.shape[0] == batch_size
```

## ML/DL Code Quality Checklist

Before marking work complete:
- [ ] Seeds set for reproducibility (torch, numpy, random)
- [ ] Device-agnostic code (no hardcoded CUDA/MPS)
- [ ] Safe serialization (weights_only=True or safetensors)
- [ ] Checkpointing includes config and metrics
- [ ] Training config uses dataclasses (not dicts)
- [ ] ML/DL-specific error handling (CheckpointNotFoundError, etc.)
- [ ] Tests for data pipelines and model forward passes
- [ ] Reproducibility documented (seeds, deterministic flags)
- [ ] ruff check passes
- [ ] pyright passes
- [ ] Tests written and passing

> **Note**: For general code quality checklist (type hints, pathlib, logging, etc.), see project rules in `.cursor/rules/python-coding-style.md` or the rules in this library.

**Remember**: Clean, reproducible ML/DL code enables rapid experimentation and confident deployment.
