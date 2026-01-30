"""Training script for ML model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model initialization
- Training loop with checkpointing
- Validation and metrics tracking
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults."""

    # Model
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 10
    num_workers: int = 4

    # Paths
    data_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("models/checkpoints")

    # Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Advanced
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Dictionary of training metrics
    """
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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: Device to run on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs

        total_loss += loss.item() * batch.get("labels", batch.get("input_ids")).size(0)
        total_samples += batch.get("labels", batch.get("input_ids")).size(0)

    return {"loss": total_loss / total_samples}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def main() -> None:
    """Main training function."""
    config = TrainingConfig()

    # Set seed for reproducibility
    set_seed(config.seed)

    # Setup device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Initialize model
    # model = create_model(config)

    # TODO: Load datasets
    # train_loader, val_loader = create_dataloaders(config)

    # TODO: Initialize optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(config.max_epochs):
        # TODO: Uncomment when model/dataloader are ready
        # train_metrics = train_epoch(model, train_loader, optimizer, device)
        # val_metrics = evaluate(model, val_loader, device)

        # logger.info(
        #     f"Epoch {epoch + 1}/{config.max_epochs} - "
        #     f"Train Loss: {train_metrics['loss']:.4f} - "
        #     f"Val Loss: {val_metrics['loss']:.4f}"
        # )

        # Save checkpoint
        # save_checkpoint(
        #     model,
        #     optimizer,
        #     epoch,
        #     val_metrics,
        #     config.checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
        # )

        logger.info(f"Epoch {epoch + 1}/{config.max_epochs} completed (placeholder)")

    logger.info("Training completed!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
