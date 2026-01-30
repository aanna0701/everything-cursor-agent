---
name: ml-training-patterns
description: ML training patterns - dataset/dataloader, gradient accumulation, AMP, checkpointing, early stopping, metrics tracking.
---

# ML Training Patterns

Patterns for training deep learning models.

## Dataset & DataLoader

### Custom Dataset

```python
from pathlib import Path
from torch.utils.data import Dataset
import json


class TextDataset(Dataset):
    """Dataset for text classification."""

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, path: Path) -> list[dict]:
        """Load data from JSONL file."""
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }
```

### Custom Collate Function

```python
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for variable-length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    # Pad sequences to max length in batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

### DataLoader Setup

```python
from torch.utils.data import DataLoader, random_split


def create_dataloaders(
    dataset: Dataset,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    # Split dataset
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
```

## Training Loop

### Basic Training Loop

```python
import logging
import torch
from torch.utils.data import DataLoader

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
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = torch.nn.functional.cross_entropy(outputs, batch["labels"])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch["labels"].size(0)
        predictions = outputs.argmax(dim=-1)
        total_correct += (predictions == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
```

### Validation Loop

```python
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = torch.nn.functional.cross_entropy(outputs, batch["labels"])

        total_loss += loss.item() * batch["labels"].size(0)
        predictions = outputs.argmax(dim=-1)
        total_correct += (predictions == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
```

## Gradient Accumulation

```python
def train_epoch_with_accumulation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 4,
) -> dict[str, float]:
    """Train with gradient accumulation for effective larger batch size."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass (scale loss by accumulation steps)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = torch.nn.functional.cross_entropy(outputs, batch["labels"])
        loss = loss / accumulation_steps

        # Backward pass (accumulate gradients)
        loss.backward()

        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * batch["labels"].size(0)
        total_samples += batch["labels"].size(0)

    # Handle remaining gradients
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return {"loss": total_loss / total_samples}
```

## Mixed Precision (AMP)

```python
from torch.amp import GradScaler, autocast


def train_epoch_with_amp(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
) -> dict[str, float]:
    """Train with automatic mixed precision."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast(device_type="cuda"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = torch.nn.functional.cross_entropy(outputs, batch["labels"])

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Unscale gradients and clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch["labels"].size(0)
        total_samples += batch["labels"].size(0)

    return {"loss": total_loss / total_samples}


# Setup
scaler = GradScaler("cuda")
train_epoch_with_amp(model, train_loader, optimizer, device, scaler)
```

## Gradient Clipping

```python
def train_step_with_clipping(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
) -> float:
    """Single training step with gradient clipping."""
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    optimizer.step()

    return loss.item()
```

## Checkpointing

```python
from pathlib import Path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def save_best_model(
    model: torch.nn.Module,
    metrics: dict[str, float],
    best_metric: float,
    metric_name: str,
    checkpoint_dir: Path,
    mode: str = "min",
) -> float:
    """Save model if it's the best so far."""
    current = metrics[metric_name]

    is_best = (mode == "min" and current < best_metric) or (
        mode == "max" and current > best_metric
    )

    if is_best:
        torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        return current

    return best_metric
```

## Early Stopping

```python
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Early stopping handler."""

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    counter: int = 0
    best_score: float | None = None
    should_stop: bool = False

    def __call__(self, metric: float) -> bool:
        """Check if training should stop.

        Args:
            metric: Current validation metric.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "min":
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# Usage
early_stopping = EarlyStopping(patience=5, mode="min")

for epoch in range(max_epochs):
    train_metrics = train_epoch(model, train_loader, optimizer, device)
    val_metrics = evaluate(model, val_loader, device)

    if early_stopping(val_metrics["loss"]):
        logger.info(f"Early stopping at epoch {epoch}")
        break
```

## Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create scheduler with warmup and cosine decay."""
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )

    decay_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_steps],
    )

    return scheduler


# Usage
scheduler = create_scheduler(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=len(train_loader) * max_epochs,
)

for epoch in range(max_epochs):
    for batch in train_loader:
        train_step(model, batch, optimizer)
        scheduler.step()
```

## Complete Training Loop

```python
def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    """Complete training loop with all features."""
    model = model.to(device)

    # Setup
    scaler = GradScaler() if config.use_amp else None
    early_stopping = EarlyStopping(patience=config.patience)
    best_val_loss = float("inf")

    for epoch in range(config.max_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Logging
        logger.info(
            f"Epoch {epoch + 1}/{config.max_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Save best model
        best_val_loss = save_best_model(
            model, val_metrics, best_val_loss, "loss",
            config.checkpoint_dir, mode="min"
        )

        # Save periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                config.checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            )

        # Early stopping
        if early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Step scheduler
        scheduler.step()

    return val_metrics
```

## Metrics Tracking

```python
from collections import defaultdict


class MetricTracker:
    """Track metrics during training."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        """Add metrics to history."""
        for name, value in metrics.items():
            self.history[name].append(value)

    def get_best(self, metric: str, mode: str = "min") -> tuple[int, float]:
        """Get best epoch and value for a metric."""
        values = self.history[metric]
        if mode == "min":
            best_idx = min(range(len(values)), key=lambda i: values[i])
        else:
            best_idx = max(range(len(values)), key=lambda i: values[i])
        return best_idx + 1, values[best_idx]

    def summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, values in self.history.items():
            summary[name] = {
                "min": min(values),
                "max": max(values),
                "last": values[-1],
            }
        return summary
```

## Checklist

Before starting training:
- [ ] Dataset and DataLoader working
- [ ] Model on correct device
- [ ] Optimizer and scheduler configured
- [ ] Mixed precision enabled (if GPU supports it)
- [ ] Gradient clipping set
- [ ] Checkpointing configured
- [ ] Early stopping configured
- [ ] Metrics logging set up
- [ ] Seeds set for reproducibility

**Remember**: Training stability comes from proper gradient handling, learning rate scheduling, and regular checkpointing.
