---
name: tdd-workflow
description: Use this skill when writing new features, fixing bugs, or refactoring ML/DL code. Enforces pytest-based test-driven development with focus on data pipelines, model wrappers, and training smoke tests.
---

# Test-Driven Development for ML/DL

Pytest-based TDD workflow for machine learning and deep learning projects.

## When to Activate

- Writing new data processing pipelines
- Implementing model architectures
- Creating training loops
- Building inference services
- Fixing bugs in ML code
- Refactoring existing ML code

## Core Principles

### 1. Tests BEFORE Code
Write tests first, then implement code to make tests pass.

### 2. Focus on What's Testable
- Data pipelines: deterministic, highly testable
- Model wrappers: input/output shapes, forward passes
- Preprocessing: transforms, augmentations
- Training smoke tests: 1-2 steps with tiny data
- Avoid testing non-deterministic training outcomes

### 3. Coverage Guidance
- Aim for 80%+ on data/preprocessing code
- Focus on critical paths, not training internals
- Use tolerances for floating-point comparisons
- Don't force E2E tests if unit tests suffice

## TDD Workflow Steps

### Step 1: Define What You're Testing

```python
# User story format
# As a data scientist, I want to preprocess images consistently,
# so that model training is reproducible.
```

### Step 2: Write Failing Test First (RED)

```python
# tests/test_transforms.py
import pytest
import torch
from project.data.transforms import ImageTransform


class TestImageTransform:
    @pytest.fixture
    def transform(self) -> ImageTransform:
        return ImageTransform(size=224, normalize=True)

    def test_output_shape(self, transform: ImageTransform) -> None:
        """Transform produces correct output shape."""
        # Arrange
        image = torch.randn(3, 256, 256)

        # Act
        output = transform(image)

        # Assert
        assert output.shape == (3, 224, 224)

    def test_normalized_range(self, transform: ImageTransform) -> None:
        """Normalized output is in expected range."""
        image = torch.rand(3, 256, 256) * 255

        output = transform(image)

        # ImageNet normalization typically in [-2.5, 2.5]
        assert output.min() >= -3.0
        assert output.max() <= 3.0
```

### Step 3: Run Test - Verify It Fails

```bash
pytest tests/test_transforms.py -v
# Should fail - ImageTransform not implemented yet
```

### Step 4: Write Minimal Implementation (GREEN)

```python
# src/project/data/transforms.py
import torch
import torch.nn.functional as F


class ImageTransform:
    """Image preprocessing transform."""

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, size: int = 224, normalize: bool = True) -> None:
        self.size = size
        self.normalize = normalize

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Resize
        image = F.interpolate(
            image.unsqueeze(0), size=(self.size, self.size), mode="bilinear"
        ).squeeze(0)

        # Normalize
        if self.normalize:
            image = image / 255.0
            image = (image - self.IMAGENET_MEAN[:, None, None]) / self.IMAGENET_STD[:, None, None]

        return image
```

### Step 5: Run Test - Verify It Passes

```bash
pytest tests/test_transforms.py -v
# Should pass
```

### Step 6: Refactor (IMPROVE)
Improve code while keeping tests green.

### Step 7: Verify Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

## ML/DL Testing Patterns

### Unit Tests: Data Pipelines

```python
import pytest
import torch
from torch.utils.data import DataLoader

from project.data.dataset import TextDataset
from project.data.collate import collate_fn


class TestTextDataset:
    @pytest.fixture
    def dataset(self, tmp_path) -> TextDataset:
        # Create minimal test data
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"text": "hello world", "label": 0}\n')
        return TextDataset(data_file)

    def test_len(self, dataset: TextDataset) -> None:
        assert len(dataset) == 1

    def test_getitem_returns_dict(self, dataset: TextDataset) -> None:
        item = dataset[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "label" in item

    def test_getitem_shapes(self, dataset: TextDataset) -> None:
        item = dataset[0]
        assert item["input_ids"].dim() == 1
        assert item["label"].dim() == 0

    def test_dataloader_batching(self, dataset: TextDataset) -> None:
        """Dataset works with DataLoader."""
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 1


class TestCollateFn:
    def test_pads_sequences(self) -> None:
        """Collate function pads variable-length sequences."""
        samples = [
            {"input_ids": torch.tensor([1, 2, 3]), "label": torch.tensor(0)},
            {"input_ids": torch.tensor([4, 5]), "label": torch.tensor(1)},
        ]

        batch = collate_fn(samples)

        assert batch["input_ids"].shape == (2, 3)  # Padded to max length
        assert batch["labels"].shape == (2,)
```

### Unit Tests: Model Forward Pass

```python
import pytest
import torch

from project.models import TextClassifier


class TestTextClassifier:
    @pytest.fixture
    def model(self) -> TextClassifier:
        return TextClassifier(
            vocab_size=1000,
            embed_dim=64,
            num_classes=3,
        )

    def test_forward_shape(self, model: TextClassifier) -> None:
        """Forward pass returns correct shape."""
        input_ids = torch.randint(0, 1000, (4, 32))  # (batch, seq_len)

        logits = model(input_ids)

        assert logits.shape == (4, 3)  # (batch, num_classes)

    def test_forward_no_nan(self, model: TextClassifier) -> None:
        """Forward pass produces no NaN values."""
        input_ids = torch.randint(0, 1000, (2, 16))

        logits = model(input_ids)

        assert not torch.isnan(logits).any()

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_variable_batch_size(
        self, model: TextClassifier, batch_size: int
    ) -> None:
        """Model handles variable batch sizes."""
        input_ids = torch.randint(0, 1000, (batch_size, 16))

        logits = model(input_ids)

        assert logits.shape[0] == batch_size

    def test_eval_mode_deterministic(self, model: TextClassifier) -> None:
        """Model is deterministic in eval mode."""
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 16))

        out1 = model(input_ids)
        out2 = model(input_ids)

        torch.testing.assert_close(out1, out2)
```

### Training Smoke Test (CRITICAL)

```python
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from project.models import TextClassifier
from project.training import Trainer, TrainingConfig


class TestTrainingSmokeTest:
    """Minimal training tests to verify wiring."""

    @pytest.fixture
    def tiny_dataloader(self) -> DataLoader:
        """Create tiny dataset for smoke testing."""
        # 8 samples, sequence length 16
        input_ids = torch.randint(0, 100, (8, 16))
        labels = torch.randint(0, 3, (8,))
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=4)

    @pytest.fixture
    def model(self) -> TextClassifier:
        return TextClassifier(vocab_size=100, embed_dim=32, num_classes=3)

    def test_single_training_step(
        self, model: TextClassifier, tiny_dataloader: DataLoader
    ) -> None:
        """Model can complete one training step without error."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = next(iter(tiny_dataloader))
        input_ids, labels = batch

        # Forward
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        assert loss.item() > 0  # Loss should be positive
        assert not torch.isnan(loss)

    def test_loss_decreases_over_steps(
        self, model: TextClassifier, tiny_dataloader: DataLoader
    ) -> None:
        """Loss decreases after a few training steps (sanity check)."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []

        for _ in range(5):  # Just 5 steps
            for batch in tiny_dataloader:
                input_ids, labels = batch
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Loss should generally decrease (allow some variance)
        assert losses[-1] < losses[0] * 1.5  # Not strictly decreasing

    def test_checkpoint_save_load(
        self, model: TextClassifier, tmp_path
    ) -> None:
        """Model can be saved and loaded."""
        # Save
        checkpoint_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Load into new model
        new_model = TextClassifier(vocab_size=100, embed_dim=32, num_classes=3)
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Verify same outputs
        model.eval()
        new_model.eval()
        x = torch.randint(0, 100, (2, 16))
        torch.testing.assert_close(model(x), new_model(x))
```

### Integration Tests: Metrics

```python
import pytest
import torch

from project.training.metrics import compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        """Metrics are correct for perfect predictions."""
        predictions = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 0, 1])

        metrics = compute_metrics(predictions, labels)

        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_random_predictions(self) -> None:
        """Metrics handle random predictions."""
        predictions = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 1, 2, 3])

        metrics = compute_metrics(predictions, labels)

        assert metrics["accuracy"] == pytest.approx(0.25)

    def test_floating_point_tolerance(self) -> None:
        """Use tolerance for floating-point comparisons."""
        predictions = torch.tensor([0, 1])
        labels = torch.tensor([0, 1])

        metrics = compute_metrics(predictions, labels)

        # Use pytest.approx for floating-point
        assert metrics["accuracy"] == pytest.approx(1.0, rel=1e-5)
```

## Mocking External Services

### Mock HuggingFace Model

```python
import pytest
from unittest.mock import MagicMock, patch
import torch


@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    tokenizer.pad_token_id = 0
    return tokenizer


def test_with_mock_tokenizer(mock_tokenizer):
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        # Test code that uses tokenizer
        pass
```

### Mock GPU/CUDA

```python
import pytest
import torch


@pytest.fixture
def force_cpu(monkeypatch):
    """Force CPU for testing even if GPU available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_cpu_fallback(force_cpu):
    """Test runs on CPU."""
    from project.utils import get_device

    device = get_device()
    assert device.type == "cpu"
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_transforms.py   # Data transforms
│   ├── test_dataset.py      # Dataset classes
│   ├── test_model.py        # Model forward pass
│   └── test_metrics.py      # Metric functions
├── integration/
│   ├── test_pipeline.py     # Data pipeline end-to-end
│   └── test_training.py     # Training smoke tests
└── fixtures/
    └── sample_data/         # Minimal test data
```

## conftest.py Example

```python
# tests/conftest.py
import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get test device (prefer CPU for speed/reproducibility)."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_seed():
    """Set seeds before each test for reproducibility."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def tiny_model():
    """Create tiny model for fast testing."""
    from project.models import TextClassifier

    return TextClassifier(vocab_size=100, embed_dim=16, num_classes=2)
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_model.py

# Run tests matching pattern
pytest -k "training"

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Common Mistakes to Avoid

### WRONG: Testing Training Convergence

```python
# DON'T: This is flaky and slow
def test_model_converges():
    # Train for 100 epochs
    # Assert final accuracy > 0.9
    # This is NOT a unit test
```

### CORRECT: Test Training Mechanics

```python
# DO: Test that training runs without error
def test_training_step_runs():
    # One forward + backward pass
    # Assert no errors, loss is finite
```

### WRONG: Exact Floating-Point Comparison

```python
# DON'T: Will fail due to floating-point precision
assert output == 0.5
```

### CORRECT: Use Tolerances

```python
# DO: Use appropriate tolerance
assert output == pytest.approx(0.5, rel=1e-5)
torch.testing.assert_close(tensor1, tensor2, rtol=1e-4, atol=1e-4)
```

### WRONG: Testing Random Behavior

```python
# DON'T: Non-deterministic test
def test_dropout_output():
    model.train()  # Dropout active
    assert model(x) == expected  # Will fail randomly
```

### CORRECT: Control Randomness

```python
# DO: Test in eval mode or set seed
def test_model_eval():
    model.eval()  # Dropout disabled
    torch.testing.assert_close(model(x), model(x))
```

## Success Metrics

- Data pipelines: 90%+ coverage
- Model forward pass: tested for all input shapes
- Training: smoke test passes (1-2 steps)
- Metrics: all edge cases covered
- Tests run in < 30 seconds total
- No flaky tests

**Remember**: Focus on testing what's deterministic and critical. Training outcomes are inherently stochastic; test the mechanics, not the results.
