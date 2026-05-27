from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets.dataset import BandGapDataset, BandGapScaler
from src.models.INN import INN


TRAIN_CSV = PROJECT_ROOT / "data" / "generated_data" / "train_data.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "inn_dim8_best.pth"
FEATURE_NAMES = ["E", "nu", "rho", "h", "a", "mR", "fR_target"]

def find_band_gaps(freqs: np.ndarray, k_slice = None, fmin: float | None = None, fmax: float | None = None, min_width: float = 1e-6):
    arr = freqs if k_slice is None else freqs[k_slice]

    band_mins = np.min(arr, axis=0)
    band_maxs = np.max(arr, axis=0)

    gaps = []
    for i in range(arr.shape[1] - 1):
        low = band_maxs[i]
        high = band_mins[i + 1]
        width = high - low

        if width > min_width:
            low_clip = max(low, fmin) if fmin is not None else low
            high_clip = min(high, fmax) if fmax is not None else high

            if high_clip - low_clip > min_width:
                gaps.append({
                    "band_left": i,
                    "band_right": i + 1,
                    "f_low": float(low_clip),
                    "f_high": float(high_clip),
                    "width": float(high_clip - low_clip),
                })

    gaps.sort(key=lambda g: g["width"], reverse=True)
    return gaps

def get_band_gap(
    E: float = 70e9,
    nu: float = 0.3,
    rho: float = 2700,
    h: float = 0.002,
    a: float = 0.1,
    mR: float = 0.027,
    fR_target: float = 300,
    M_order: int = 5):

    kR = mR * (2 * np.pi * fR_target)**2 

    D_const = (E * h**3) / (12 * (1 - nu**2))
    S = a**2          
    mass_plate = rho * h * S

    range_M = np.arange(-M_order, M_order + 1)
    GX, GY = np.meshgrid(range_M, range_M)
    GX, GY = GX.flatten() * (2 * np.pi / a), GY.flatten() * (2 * np.pi / a)
    N_waves = len(GX)

    
    pts = 40
    path_k = np.zeros((3 * pts, 2))
    path_k[0:pts, 0] = np.linspace(0, np.pi/a, pts)
    path_k[pts:2*pts, 0] = np.pi/a
    path_k[pts:2*pts, 1] = np.linspace(0, np.pi/a, pts)
    path_k[2*pts:, 0] = np.linspace(np.pi/a, 0, pts)
    path_k[2*pts:, 1] = np.linspace(np.pi/a, 0, pts)

    freqs_lr = []
    freqs_bare = []

    for kx, ky in path_k:
        K_diag = ((kx + GX)**2 + (ky + GY)**2)**2
        K_mat = np.diag(K_diag)
    
        U = np.ones((N_waves, N_waves))
        I = np.eye(N_waves)

        LHS_b = D_const * S * K_mat
        RHS_b = mass_plate * I
        vals_b, _ = eigh(LHS_b, RHS_b)
        freqs_bare.append(np.sqrt(np.maximum(vals_b, 0)) / (2 * np.pi))

        A = np.zeros((N_waves + 1, N_waves + 1))
        A[:N_waves, :N_waves] = D_const * S * K_mat + kR * U
        A[:N_waves, N_waves] = -kR * np.ones(N_waves)
        A[N_waves, :N_waves] = -kR * np.ones(N_waves)
        A[N_waves, N_waves] = kR
        
        B = np.zeros((N_waves + 1, N_waves + 1))
        B[:N_waves, :N_waves] = mass_plate * I
        B[N_waves, N_waves] = mR
        
        vals_lr, _ = eigh(A, B)
        freqs_lr.append(np.sqrt(np.maximum(vals_lr, 0)) / (2 * np.pi))

    freqs_lr = np.array(freqs_lr)
    freqs_bare = np.array(freqs_bare)

    gaps_all = find_band_gaps(freqs_lr, fmin=0, fmax=800)

    return gaps_all

def get_dispersion_plot(
    E: float = 70e9,
    nu: float = 0.3,
    rho: float = 2700,
    h: float = 0.002,
    a: float = 0.1,
    mR: float = 0.027,
    fR_target: float = 300,
    M_order: int = 5, 
    title: str | None = None,
    params: dict[str, float] | None = None,
    f_low: float | None = None,
    f_high: float | None = None
) -> None:
    kR = mR * (2 * np.pi * fR_target) ** 2

    D_const = (E * h**3) / (12 * (1 - nu**2))
    S = a**2
    mass_plate = rho * h * S

    range_M = np.arange(-M_order, M_order + 1)
    GX, GY = np.meshgrid(range_M, range_M)
    GX = GX.flatten() * (2 * np.pi / a)
    GY = GY.flatten() * (2 * np.pi / a)
    N_waves = len(GX)

    pts = 40
    path_k = np.zeros((3 * pts, 2))
    path_k[0:pts, 0] = np.linspace(0, np.pi / a, pts)
    path_k[pts : 2 * pts, 0] = np.pi / a
    path_k[pts : 2 * pts, 1] = np.linspace(0, np.pi / a, pts)
    path_k[2 * pts :, 0] = np.linspace(np.pi / a, 0, pts)
    path_k[2 * pts :, 1] = np.linspace(np.pi / a, 0, pts)

    freqs_lr = []
    freqs_bare = []

    for kx, ky in path_k:
        K_diag = ((kx + GX) ** 2 + (ky + GY) ** 2) ** 2
        K_mat = np.diag(K_diag)
        U = np.ones((N_waves, N_waves))
        I = np.eye(N_waves)

        LHS_b = D_const * S * K_mat
        RHS_b = mass_plate * I
        vals_b, _ = eigh(LHS_b, RHS_b)
        freqs_bare.append(np.sqrt(np.maximum(vals_b, 0.0)) / (2 * np.pi))

        A = np.zeros((N_waves + 1, N_waves + 1))
        A[:N_waves, :N_waves] = D_const * S * K_mat + kR * U
        A[:N_waves, N_waves] = -kR * np.ones(N_waves)
        A[N_waves, :N_waves] = -kR * np.ones(N_waves)
        A[N_waves, N_waves] = kR

        B = np.zeros((N_waves + 1, N_waves + 1))
        B[:N_waves, :N_waves] = mass_plate * I
        B[N_waves, N_waves] = mR

        vals_lr, _ = eigh(A, B)
        freqs_lr.append(np.sqrt(np.maximum(vals_lr, 0.0)) / (2 * np.pi))

    freqs_lr = np.array(freqs_lr)
    freqs_bare = np.array(freqs_bare)

    plt.figure(figsize=(10, 7))

    for i in range(8):
        label = "Bare plate" if i == 0 else ""
        plt.scatter(range(len(path_k)), freqs_bare[:, i], color="blue", s=2, alpha=0.5, label=label)

    for i in range(8):
        label = "LR plate" if i == 0 else ""
        plt.plot(freqs_lr[:, i], color="black", linewidth=1.5, label=label)

    plt.xlim(0, len(path_k) - 1)
    plt.ylim(0, 800)
    plt.ylabel("Frequency (Hz)", fontsize=14)
    if title is None:
        plt.title(f"Band Structure (f_R = {fR_target:.3f} Hz)", fontsize=16)
    else:
        plt.title(f"{title} Band Structure", fontsize=16)
    plt.axvline(pts, color="gray", linestyle="--")
    plt.axvline(2 * pts, color="gray", linestyle="--")
    plt.xticks([0, pts, 2 * pts, 3 * pts - 1], ["$\\Gamma$", "X", "M", "$\\Gamma$"], fontsize=14)

    if params is not None:
        params_text = "\n".join([
            f"E = {params['E'] / 1e9:.2f} GPa",
            f"nu = {params['nu']:.4f}",
            f"rho = {params['rho']:.1f} kg/m^3",
            f"h = {params['h'] * 1e3:.4f} mm",
            f"a = {params['a']:.4f} m",
            f"mR = {params['mR']:.4f} kg",
            f"fR = {params['fR_target']:.2f} Hz",
        ])

        plt.gca().text(
            0.02,
            0.98,
            params_text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    if f_low is not None:
        plt.axhline(f_low, color="red", linestyle="--", linewidth=1.5)

    if f_high is not None:
        plt.axhline(f_high, color="red", linestyle="--", linewidth=1.5, label="band gap querry")


    plt.legend(loc="upper right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def _load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[INN, dict, BandGapScaler]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    scaler_state = checkpoint.get("scaler_state_dict")

    if scaler_state is not None:
        scaler = BandGapScaler.from_state_dict(scaler_state)

    model = INN(
        dim=8,
        num_blocks=config.get("num_blocks", 6),
        hidden_dim=config.get("hidden_dim", 128),
        hidden_layers=config.get("hidden_layers", 4),
        clamp_scale=config.get("clamp_scale", 2.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, scaler


def _make_inverse_target(f_low: float, f_high: float, device: torch.device) -> torch.Tensor:
    gap = torch.tensor([[f_low, f_high]], dtype=torch.float32, device=device)
    return gap.repeat(1, 4)


def predict_parameters_from_band_gap(
    f_low: float,
    f_high: float,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str | None = None,
) -> dict[str, float]:
    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config, scaler = _load_checkpoint(checkpoint_path, resolved_device)

    target = _make_inverse_target(f_low, f_high, resolved_device)

    if config.get("normalize_y", False):
        scaler.y_min = scaler.y_min.to(resolved_device) if scaler.y_min is not None else None
        scaler.y_max = scaler.y_max.to(resolved_device) if scaler.y_max is not None else None
        target = scaler.transform_y(target)

    with torch.no_grad():
        x_pred = model.inverse(target)

    if config.get("normalize_X", False):
        scaler.X_min = scaler.X_min.to(resolved_device) if scaler.X_min is not None else None
        scaler.X_max = scaler.X_max.to(resolved_device) if scaler.X_max is not None else None
        x_pred = scaler.inverse_transform_X(x_pred)

    x_pred = x_pred.squeeze(0).detach().cpu().numpy()
    first_seven = x_pred[:7]
    return {
name: float(value) for name, value in zip(FEATURE_NAMES, first_seven)
}

def check_band_gaps(
    f_low: float,
    f_high: float,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str | None = None,
    title: str | None = None
) -> dict[str, float]:
    params = predict_parameters_from_band_gap(
        f_low=f_low,
        f_high=f_high,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    gap = get_band_gap(**params)[0]

    print("Predicted parameters from inverse INN:")
    for key in FEATURE_NAMES:
        print(f"  {key}: {params[key]}")

    get_dispersion_plot(
        E=params["E"],
        nu=params["nu"],
        rho=params["rho"],
        h=params["h"],
        a=params["a"],
        mR=params["mR"],
        fR_target=params["fR_target"],
        title=title, 
        params=params,
        f_low=gap["f_low"],
        f_high=gap["f_high"]
    )

    return params


if __name__ == "__main__":
    # check_band_gaps(f_low=30.0, f_high=40.0, title="Querry band gap (30hz, 40hz)")
    check_band_gaps(f_low=150.0, f_high=200.0,  title="Querry band gap (150hz, 200hz)")
