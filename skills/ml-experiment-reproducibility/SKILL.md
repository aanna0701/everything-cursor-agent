---
name: ml-experiment-reproducibility
description: ML experiment reproducibility patterns - seeds, determinism, config management, artifact layout, run IDs, model cards.
---

# ML Experiment Reproducibility

Patterns for reproducible ML experiments.

## Seed Management

### Setting Seeds

```python
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enable deterministic algorithms.

    Note:
        For full CUDA determinism, set env var:
        CUBLAS_WORKSPACE_CONFIG=:4096:8
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # May impact performance, enables determinism checking
        torch.use_deterministic_algorithms(True, warn_only=True)


# Usage: Call at the start of training
set_seed(42)
```

### Determinism Caveats

```python
# Operations with non-deterministic behavior:
# - torch.nn.functional.interpolate (some modes)
# - torch.nn.functional.grid_sample
# - torch.scatter_add
# - torch.index_add
# - Sparse-dense matrix operations

# Document when you cannot achieve full determinism:
"""
Note: This training run uses grid_sample which has non-deterministic
behavior on CUDA. Results may vary slightly between runs even with
the same seed. For exact reproduction, run on CPU.
"""
```

## Configuration Management

### Dataclass Config

```python
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Experiment
    experiment_name: str = "baseline"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    seed: int = 42

    # Data
    data_path: Path = field(default_factory=lambda: Path("data"))
    train_split: float = 0.8
    val_split: float = 0.1

    # Model
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    num_layers: int = 12
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)

    def save(self, path: Path) -> None:
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
```

### Config with Hydra/OmegaConf (Optional)

```yaml
# configs/experiment/base.yaml
experiment_name: baseline
seed: 42

data:
  path: data/
  train_split: 0.8
  val_split: 0.1

model:
  name: bert-base-uncased
  hidden_size: 768
  num_layers: 12
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 1e-4
  max_epochs: 10
```

```python
# Using Hydra
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="experiment/base")
def train(cfg: DictConfig) -> None:
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Learning rate: {cfg.training.learning_rate}")
```

## Artifact Layout

### Directory Structure

```
outputs/
├── experiments/
│   └── {experiment_name}/
│       └── {run_id}/
│           ├── config.json           # Full configuration
│           ├── checkpoints/
│           │   ├── best.pt           # Best model
│           │   ├── last.pt           # Latest checkpoint
│           │   └── epoch_005.pt      # Periodic checkpoints
│           ├── logs/
│           │   ├── train.log         # Training logs
│           │   ├── metrics.jsonl     # Metrics per step/epoch
│           │   └── tensorboard/      # TensorBoard logs
│           ├── model_card.md         # Model documentation
│           └── results/
│               ├── predictions.json  # Eval predictions
│               └── metrics.json      # Final metrics
```

### Creating Experiment Directory

```python
from pathlib import Path
from datetime import datetime


def create_experiment_dir(
    base_dir: Path,
    experiment_name: str,
    run_id: str | None = None,
) -> Path:
    """Create experiment directory structure.

    Args:
        base_dir: Base output directory.
        experiment_name: Name of the experiment.
        run_id: Optional run ID (defaults to timestamp).

    Returns:
        Path to the run directory.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = base_dir / "experiments" / experiment_name / run_id

    # Create subdirectories
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    return run_dir


# Usage
run_dir = create_experiment_dir(Path("outputs"), "text_classification")
config.save(run_dir / "config.json")
```

## Run Tracking

### Metrics Logging

```python
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class MetricEntry:
    """Single metric log entry."""

    timestamp: str
    epoch: int
    step: int
    metrics: dict[str, float]


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, epoch: int, step: int, metrics: dict[str, float]) -> None:
        """Log metrics to JSONL file."""
        entry = MetricEntry(
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            step=step,
            metrics=metrics,
        )
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")


# Usage
logger = MetricsLogger(run_dir / "logs" / "metrics.jsonl")
logger.log(epoch=1, step=100, metrics={"loss": 0.5, "accuracy": 0.8})
```

### Integration with W&B/MLflow (Optional)

```python
# Weights & Biases
import wandb

wandb.init(
    project="my-project",
    name=config.run_id,
    config=asdict(config),
)
wandb.log({"loss": 0.5, "accuracy": 0.8})


# MLflow
import mlflow

mlflow.set_experiment(config.experiment_name)
with mlflow.start_run(run_name=config.run_id):
    mlflow.log_params(asdict(config))
    mlflow.log_metrics({"loss": 0.5, "accuracy": 0.8})
```

## Model Cards

### Template

```markdown
# Model Card: {model_name}

## Model Details

- **Model type**: Text classification (BERT-based)
- **Training date**: 2024-01-15
- **Version**: 1.0.0
- **Framework**: PyTorch 2.1

## Intended Use

- **Primary use**: Sentiment analysis for product reviews
- **Out-of-scope**: Not suitable for medical or legal text

## Training Data

- **Dataset**: Amazon product reviews (100k samples)
- **Train/Val/Test split**: 80/10/10
- **Preprocessing**: Lowercase, tokenization with BERT tokenizer

## Training Configuration

```json
{
  "batch_size": 32,
  "learning_rate": 1e-4,
  "epochs": 10,
  "seed": 42
}
```

## Evaluation Results

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.92  |
| F1 Score  | 0.91  |
| Precision | 0.90  |
| Recall    | 0.93  |

## Limitations

- Trained on English text only
- May not generalize to informal language (slang, typos)
- Performance degrades on texts > 512 tokens

## Ethical Considerations

- May reflect biases present in training data
- Not audited for fairness across demographic groups

## How to Use

```python
from project import TextClassifier

model = TextClassifier.from_pretrained("outputs/experiments/baseline/best")
prediction = model.predict("Great product!")
```
```

### Generate Model Card

```python
def generate_model_card(
    config: ExperimentConfig,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Generate model card from config and metrics."""
    template = f"""# Model Card: {config.experiment_name}

## Model Details

- **Model type**: {config.model_name}
- **Training date**: {datetime.now().strftime("%Y-%m-%d")}
- **Framework**: PyTorch {torch.__version__}

## Training Configuration

```json
{json.dumps(asdict(config), indent=2, default=str)}
```

## Evaluation Results

| Metric | Value |
|--------|-------|
""" + "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())

    output_path.write_text(template)
```

## Checkpoint Management

### Save with Full State

```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    config: ExperimentConfig,
    path: Path,
) -> None:
    """Save complete training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": asdict(config),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)
```

### Load and Resume

```python
def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load checkpoint and restore state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG states for exact reproduction
    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    return checkpoint
```

## Environment Reproducibility

### Requirements File

```bash
# Generate exact requirements
uv pip freeze > requirements.txt

# Or with uv lock
uv lock
```

### pyproject.toml Dependencies

```toml
[project]
dependencies = [
    "torch>=2.1.0,<3.0.0",
    "transformers>=4.35.0,<5.0.0",
    "numpy>=1.24.0,<2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "pyright>=1.1.0",
]
```

### Docker for Full Reproducibility

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

# Copy code
COPY . .

# Set environment for reproducibility
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONHASHSEED=0

CMD ["python", "scripts/train.py"]
```

## Checklist

Before finalizing an experiment:

- [ ] Seeds set and documented
- [ ] Config saved as JSON/YAML
- [ ] Checkpoints saved (best + last)
- [ ] Metrics logged to file
- [ ] Model card created
- [ ] Requirements/lock file updated
- [ ] Run can be reproduced from config

**Remember**: If you can't reproduce it, you can't trust it.
