from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping (dictionary).")
    return cfg


def to_float_tensor_2d(x: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.tensor(x, dtype=torch.float32, device=device)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    return t


def save_csv(path: str, header: List[str], rows: Iterable[Iterable[Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_history_csv(path: str, history: Dict[str, List[float]]) -> None:
    keys = list(history.keys())
    n = len(history[keys[0]]) if keys else 0
    rows = []
    for i in range(n):
        rows.append([history[k][i] for k in keys])
    save_csv(path, keys, rows)


def plot_history(
    history: Dict[str, List[float]],
    outpath: str,
    title: str,
    yscale: Optional[str] = "log",
) -> None:
    ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(7, 3))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel("Value")
    if yscale:
        plt.yscale(yscale)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_prediction(
    t: np.ndarray,
    u_exact: np.ndarray,
    u_pred: np.ndarray,
    outpath: str,
    title: str,
    t_obs: Optional[np.ndarray] = None,
    u_obs: Optional[np.ndarray] = None,
) -> None:
    ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(7, 3))
    if t_obs is not None and u_obs is not None:
        plt.scatter(t_obs, u_obs, s=18, alpha=0.6, label="Noisy observations")
    plt.plot(t, u_exact, label="Exact solution", alpha=0.7)
    plt.plot(t, u_pred, label="PINN solution")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
