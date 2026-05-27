from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets.dataset import BandGapDataset, BandGapScaler
from src.models.INN import INN


TRAIN_CSV = PROJECT_ROOT / "data" / "generated_data" / "train_data.csv"
VAL_CSV = PROJECT_ROOT / "data" / "generated_data" / "val_data.csv"
TEST_CSV = PROJECT_ROOT / "data" / "generated_data" / "test_data.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "inn_dim8_best2.pth"


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0
    hidden_dim: int = 128
    hidden_layers: int = 4
    num_blocks: int = 6
    clamp_scale: float = 2.0
    normalize_X: bool = True
    normalize_y: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(
    csv_path: Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    normalize_X: bool,
    normalize_y: bool,
    scaler: BandGapScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[BandGapDataset, DataLoader]:
    if fit_scaler:
        if scaler is not None:
            raise ValueError("Scaler must be None when fit_scaler=True.")
        scaler = BandGapScaler(
            normalize_X=normalize_X,
            normalize_y=normalize_y,
        )

    dataset = BandGapDataset(
        csv_path=csv_path,
        add_zero_feature=True,
        target_repeat=4,
        normalize_X=normalize_X,
        normalize_y=normalize_y,
        scaler=scaler,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        predictions = model(features)
        per_sample_mse = ((predictions - targets) ** 2).mean(dim=1)
        loss = per_sample_mse.mean()

        if training:
            loss.backward()
            optimizer.step()

        batch_size = features.size(0)
        total_loss += per_sample_mse.sum().item()
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        return run_epoch(model, loader, device, optimizer=None)


def train(config: TrainConfig) -> dict[str, float]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    train_dataset, train_loader = build_dataloader(
        TRAIN_CSV,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        normalize_X=config.normalize_X,
        normalize_y=config.normalize_y,
        fit_scaler=True,
    )
    _, val_loader = build_dataloader(
        VAL_CSV,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        normalize_X=config.normalize_X,
        normalize_y=config.normalize_y,
        scaler=train_dataset.scaler,
    )
    _, test_loader = build_dataloader(
        TEST_CSV,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        normalize_X=config.normalize_X,
        normalize_y=config.normalize_y,
        scaler=train_dataset.scaler,
    )

    model = INN(
        dim=8,
        num_blocks=config.num_blocks,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        clamp_scale=config.clamp_scale,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Model dim: 8")
    print("Loss: mean squared error averaged per sample")

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": config.__dict__,
                    "scaler_state_dict": train_dataset.scaler.state_dict(),
                },
                BEST_MODEL_PATH,
            )

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)

    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")

    return {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
    }


if __name__ == "__main__":
    train(TrainConfig())
