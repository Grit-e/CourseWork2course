from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class BandGapScaler:
    normalize_X: bool = False
    normalize_y: bool = False
    eps: float = 1e-8
    X_min: torch.Tensor | None = None
    X_max: torch.Tensor | None = None
    y_min: torch.Tensor | None = None
    y_max: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "BandGapScaler":
        if self.normalize_X:            
            self.X_min = X.min(dim=0, keepdim=True).values
            self.X_max = X.max(dim=0, keepdim=True).values
            
        if self.normalize_y:
            self.y_min = y.min(dim=0, keepdim=True).values
            self.y_max = y.max(dim=0, keepdim=True).values

        return self

    def is_fitted_for_X(self) -> bool:
        if not self.normalize_X:
            return True

        return self.X_min is not None and self.X_max is not None
        

    def is_fitted_for_y(self) -> bool:
        if not self.normalize_y:
            return True

        return self.y_min is not None and self.y_max is not None


    def is_fitted(self) -> bool:
        return self.is_fitted_for_X() and self.is_fitted_for_y()

    def transform_X(self, X: torch.Tensor) -> torch.Tensor:
        if not self.normalize_X:
            return X
        
        if self.X_min is None or self.X_max is None:
            raise ValueError("X scaler statistics are not fitted.")
        
        return (X - self.X_min) / (self.X_max - self.X_min + self.eps)


    def inverse_transform_X(self, X: torch.Tensor) -> torch.Tensor:
        if not self.normalize_X:
            return X
        
        if self.X_min is None or self.X_max is None:
            raise ValueError("X scaler statistics are not fitted.")
        
        return X * (self.X_max - self.X_min + self.eps) + self.X_min

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        if not self.normalize_y:
            return y
        
        if self.y_min is None or self.y_max is None:
            raise ValueError("y scaler statistics are not fitted.")
        
        return (y - self.y_min) / (self.y_max - self.y_min + self.eps)
    

    def inverse_transform_y(self, y: torch.Tensor) -> torch.Tensor:
        if not self.normalize_y:
            return y
        
        if self.y_min is None or self.y_max is None:
            raise ValueError("y scaler statistics are not fitted.")
        
        return y * (self.y_max - self.y_min + self.eps) + self.y_min


    def state_dict(self) -> dict[str, torch.Tensor | bool | float | None]:
        return {
            "normalize_X": self.normalize_X,
            "normalize_y": self.normalize_y,
            "eps": self.eps,
            "X_min": self.X_min,
            "X_max": self.X_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor | bool | float | None]) -> "BandGapScaler":
        return cls(
            normalize_X=bool(state_dict.get("normalize_X", False)),
            normalize_y=bool(state_dict.get("normalize_y", False)),
            eps=float(state_dict.get("eps", 1e-8)),
            X_min=state_dict.get("X_min"),
            X_max=state_dict.get("X_max"),
            y_min=state_dict.get("y_min"),
            y_max=state_dict.get("y_max"),
        )


class BandGapDataset(Dataset):
    def __init__(
        self,
        csv_path,
        feature_cols=None,
        target_cols=None,
        add_zero_feature=True,
        target_repeat=4,
        normalize_X=False,
        normalize_y=False,
        scaler: BandGapScaler | None = None,
    ):
        self.df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

        if feature_cols is None:
            feature_cols = ["E", "nu", "rho", "h", "a", "mR", "fR_target"]
        if target_cols is None:
            target_cols = ["f_low", "f_high"]

        X = self.df[feature_cols].values
        y = self.df[target_cols].values

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if add_zero_feature:
            zero_col = torch.zeros((X.shape[0], 1), dtype=torch.float32)
            X = torch.cat([X, zero_col], dim=1)

        if target_repeat > 1:
            y = y.repeat(1, target_repeat)

        self.scaler = scaler
        if self.scaler is None:
            self.scaler = BandGapScaler(
                normalize_X=normalize_X,
                normalize_y=normalize_y,
            )

        if not self.scaler.is_fitted():
            self.scaler.fit(X, y)

        X = self.scaler.transform_X(X)
        y = self.scaler.transform_y(y)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
