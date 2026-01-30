---
name: project-guidelines-example
description: Example project-specific skill for Python ML/DL - architecture, file structure, code patterns, testing, deployment. Use as template when setting up or working on ML/DL projects.
---

# Project Guidelines Skill (Example - Python ML/DL)

This is an example of a project-specific skill for Python ML/DL projects. Use this as a template for your own projects.

Based on a typical production ML pipeline: Training → Evaluation → Deployment

---

## When to Use

Reference this skill when working on the specific project it's designed for. Project skills contain:
- Architecture overview
- File structure
- Code patterns
- Testing requirements
- Deployment workflow

---

## Architecture Overview

**Tech Stack:**
- **Framework**: PyTorch / Transformers
- **Training**: Custom training loops with checkpointing
- **Inference**: FastAPI service for model serving
- **Data**: JSONL, Parquet, or custom datasets
- **Deployment**: Docker + Cloud Run / AWS Lambda / Azure Functions
- **Testing**: pytest with pytest-cov
- **Package Management**: uv (or poetry/pip-tools)

**Services:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                        │
│  PyTorch + Custom Training Loop + Checkpointing            │
│  Runs: Local GPU / Cloud (GCP/AWS)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Registry                           │
│  models/checkpoints/ (local) or MLflow / Weights & Biases  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Inference Service                        │
│  FastAPI + uvicorn + Model Loading                          │
│  Deployed: Cloud Run / ECS / Lambda                        │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation script
│   ├── inference.py          # Inference script
│   ├── models/                # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py           # Base model class
│   │   └── transformer.py    # Transformer model
│   ├── data/                  # Data loading
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset class
│   │   └── transforms.py     # Data transforms
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── config.py         # Config management
│   │   └── device.py         # Device utilities
│   └── metrics/               # Metrics
│       ├── __init__.py
│       └── accuracy.py
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_metrics.py
│
├── configs/
│   ├── train/
│   │   └── default.yaml
│   └── inference/
│       └── default.yaml
│
├── data/
│   ├── raw/                   # Original data
│   ├── processed/             # Preprocessed data
│   └── external/              # External datasets
│
├── models/
│   ├── checkpoints/           # Training checkpoints
│   ├── best/                  # Best model weights
│   └── exported/              # Exported models (ONNX, etc.)
│
├── notebooks/
│   ├── exploration/
│   └── analysis/
│
├── outputs/                   # Training outputs (logs, plots)
│
├── pyproject.toml             # Project config (uv/poetry)
├── pyrightconfig.json         # Type checking config
├── .cursorrules               # Cursor IDE rules
└── README.md
```

---

## Code Patterns

### Configuration Pattern

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults."""
    
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
    data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/checkpoints"))
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Device
    device: str = "cuda"  # Will fallback to CPU if unavailable
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
```

### Training Loop Pattern

```python
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * batch.get("labels", batch.get("input_ids")).size(0)
        total_samples += batch.get("labels", batch.get("input_ids")).size(0)
    
    return {"loss": total_loss / total_samples}
```

### Inference Service Pattern (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from pathlib import Path

app = FastAPI()

# Global model (loaded at startup)
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionRequest(BaseModel):
    """Prediction request model."""
    inputs: list[list[float]]  # Example: token IDs


class PredictionResponse(BaseModel):
    """Prediction response model."""
    predictions: list[list[float]]
    confidence: list[float]


@app.on_event("startup")
async def load_model():
    """Load model at startup."""
    global model
    checkpoint_path = Path("models/checkpoints/best.pt")
    # Load model...
    logger.info(f"Model loaded on {device}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run inference."""
    try:
        inputs = torch.tensor(request.inputs).to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.logits.softmax(dim=-1).tolist()
            confidence = outputs.logits.max(dim=-1)[0].softmax(dim=0).tolist()
        
        return PredictionResponse(
            predictions=predictions,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Testing Requirements

### Unit Tests (pytest)

```bash
# Windows PowerShell
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_models.py -v

# Run tests matching pattern
uv run pytest -k "training"
```

**Test structure:**
```python
import pytest
import torch
from src.models.transformer import TransformerModel


@pytest.fixture
def device():
    """Device fixture - always CPU for tests."""
    return torch.device("cpu")


@pytest.fixture
def model(device):
    """Model fixture."""
    model = TransformerModel(hidden_size=128, num_layers=2)
    return model.to(device)


def test_model_forward(model, device):
    """Test model forward pass."""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 128).to(device)
    
    output = model(x)
    
    assert output.shape == (batch_size, seq_len, 128)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
```

### Integration Tests (Training Smoke Test)

```python
def test_training_smoke(model, device):
    """Smoke test: training runs without error."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Tiny synthetic data
    x = torch.randn(8, 10, 128)
    y = torch.randint(0, 2, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # One training step
    model.train()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = torch.nn.functional.cross_entropy(output, batch_y)
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
        break  # Only one step for smoke test
```

---

## Deployment Workflow

### Pre-Deployment Checklist

- [ ] All tests passing locally
- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` passes
- [ ] `uv run pyright` passes
- [ ] No hardcoded secrets
- [ ] Environment variables documented
- [ ] Model checkpoints validated
- [ ] Inference service tested locally

### Deployment Commands (WSL/bash)

```bash
# Build Docker image
docker build -t ml-model-service .

# Run locally
docker run -p 8000:8000 ml-model-service

# Deploy to Cloud Run (GCP)
gcloud run deploy ml-model-service --source .

# Or deploy to AWS ECS
aws ecs create-service --cluster my-cluster --service-name ml-model-service ...
```

### Environment Variables

```bash
# .env (for local development)
MODEL_PATH=models/checkpoints/best.pt
DEVICE=cuda
LOG_LEVEL=INFO

# Production (set in deployment platform)
MODEL_PATH=gs://bucket/models/best.pt  # Or S3, Azure Blob
DEVICE=cuda
LOG_LEVEL=WARNING
API_KEY=your-secret-key
```

---

## Critical Rules

1. **Type hints** on all functions
2. **Reproducibility** - always set seeds
3. **Device placement** - explicit device management
4. **TDD** - write tests before implementation
5. **80% coverage** minimum on data/preprocessing
6. **No print()** - use logging
7. **Proper error handling** with try/except
8. **Input validation** with Pydantic
9. **Safe model loading** - use `weights_only=True` or safetensors
10. **Batch processing** - optimize DataLoader config

---

## Related Skills

- `coding-standards.md` - Python ML/DL coding best practices
- `ml-training-patterns.md` - Training loop patterns
- `ml-inference-patterns.md` - Inference patterns
- `ml-deployment-patterns.md` - Deployment patterns
- `ml-experiment-reproducibility.md` - Reproducibility guidelines
- `tdd-workflow/` - Test-driven development methodology
