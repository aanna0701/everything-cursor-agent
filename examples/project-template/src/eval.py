"""Evaluation script for ML model.

This script evaluates a trained model on test/validation datasets.
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Model
    checkpoint_path: Path = Path("models/checkpoints/best.pt")
    model_name: str = "bert-base-uncased"

    # Data
    data_dir: Path = Path("data/processed")
    test_data_path: Path = Path("data/processed/test.jsonl")
    batch_size: int = 32
    num_workers: int = 4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    output_dir: Path = Path("outputs/eval")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs

        # Calculate accuracy (if classification)
        if "labels" in batch:
            predictions = outputs.logits.argmax(dim=-1) if hasattr(outputs, "logits") else outputs.argmax(dim=-1)
            total_correct += (predictions == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)
            total_loss += loss.item() * batch["labels"].size(0)

    metrics = {"loss": total_loss / total_samples if total_samples > 0 else 0.0}
    if total_samples > 0:
        metrics["accuracy"] = total_correct / total_samples

    return metrics


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # TODO: Initialize model architecture
    # model = create_model(...)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # return model.to(device)
    raise NotImplementedError("Model loading not implemented yet")


def main() -> None:
    """Main evaluation function."""
    config = EvalConfig()

    # Setup device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {config.checkpoint_path}")
    # model = load_model(config.checkpoint_path, device)

    # TODO: Load test dataset
    # test_loader = create_dataloader(config.test_data_path, config)

    # Evaluate
    logger.info("Starting evaluation...")
    # metrics = evaluate_model(model, test_loader, device)

    # Log results
    # logger.info("Evaluation Results:")
    # for name, value in metrics.items():
    #     logger.info(f"  {name}: {value:.4f}")

    # Save results
    # results_path = config.output_dir / "results.json"
    # import json
    # with open(results_path, "w") as f:
    #     json.dump(metrics, f, indent=2)
    # logger.info(f"Results saved to {results_path}")

    logger.info("Evaluation completed (placeholder)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
