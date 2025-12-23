from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from .exact import exact_solution
from .models import FCN
from .utils import ensure_dir, load_config, plot_history, plot_prediction, save_csv, save_history_csv, set_seed


def train_inverse(cfg: Dict) -> None:
    # -------------------------
    # config
    # -------------------------
    exp = cfg["experiment"]
    results_dir = exp["results_dir"]
    ensure_dir(results_dir)

    set_seed(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cpu"))

    n_hidden = int(cfg["model"]["n_hidden"])
    n_layers = int(cfg["model"]["n_layers"])

    t0 = float(cfg["data"]["t0"])
    t1 = float(cfg["data"]["t1"])
    n_physics = int(cfg["data"]["n_physics"])
    n_test = int(cfg["data"]["n_test"])

    n_obs = int(cfg["observations"]["n_obs"])
    noise_std = float(cfg["observations"]["noise_std"])

    omega_true = float(cfg["sdof"]["omega_true"])
    xi = float(cfg["sdof"]["xi"])
    u0 = float(cfg["sdof"]["u0"])
    v0 = float(cfg["sdof"]["v0"])

    steps = int(cfg["train"]["steps"])
    lr = float(cfg["train"]["lr"])
    lambda_data = float(cfg["train"]["lambda_data"])
    plot_every = int(cfg["train"]["plot_every"])

    # -------------------------
    # noisy observations (workshop-equivalent)
    # -------------------------
    t_obs = torch.rand(n_obs, device=device).view(-1, 1)
    u_obs = exact_solution(omega_true, xi, t_obs, u0=u0, v0=v0) + noise_std * torch.randn_like(t_obs)

    # -------------------------
    # model + physics points
    # -------------------------
    pinn = FCN(1, 1, n_hidden, n_layers).to(device)
    t_physics = torch.linspace(t0, t1, n_physics, device=device).view(-1, 1).requires_grad_(True)

    # learnable omega (softplus parameterization, workshop-equivalent)
    omega_raw = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device))

    optimiser = torch.optim.Adam(list(pinn.parameters()) + [omega_raw], lr=lr)

    t_test = torch.linspace(t0, t1, n_test, device=device).view(-1, 1)
    u_exact = exact_solution(omega_true, xi, t_test, u0=u0, v0=v0)

    history: Dict[str, List[float]] = {"loss": [], "loss_phys": [], "loss_data": [], "omega": []}

    # -------------------------
    # training
    # -------------------------
    for i in range(steps):
        optimiser.zero_grad()

        omega = torch.nn.functional.softplus(omega_raw) + 1e-6
        mu = 2.0 * xi * omega
        k = omega**2

        # physics loss
        u_p = pinn(t_physics)
        dudt = torch.autograd.grad(u_p, t_physics, torch.ones_like(u_p), create_graph=True)[0]
        d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
        loss_phys = torch.mean((d2udt2 + mu * dudt + k * u_p) ** 2)

        # data loss
        u_hat = pinn(t_obs)
        loss_data = torch.mean((u_hat - u_obs) ** 2)

        loss = loss_phys + lambda_data * loss_data
        loss.backward()
        optimiser.step()

        # record
        history["loss"].append(float(loss.detach().cpu()))
        history["loss_phys"].append(float(loss_phys.detach().cpu()))
        history["loss_data"].append(float(loss_data.detach().cpu()))
        history["omega"].append(float(omega.detach().cpu()))

        # periodic plot
        if i % plot_every == 0:
            u_pred = pinn(t_test).detach().cpu().numpy().reshape(-1)
            t_np = t_test.detach().cpu().numpy().reshape(-1)
            u_ex_np = u_exact.detach().cpu().numpy().reshape(-1)
            plot_prediction(
                t=t_np,
                u_exact=u_ex_np,
                u_pred=u_pred,
                outpath=os.path.join(results_dir, f"pred_step_{i}.png"),
                title=f"Inverse PINN (step {i})",
                t_obs=t_obs.detach().cpu().numpy().reshape(-1),
                u_obs=u_obs.detach().cpu().numpy().reshape(-1),
            )

    # -------------------------
    # save artifacts
    # -------------------------
    save_history_csv(os.path.join(results_dir, "history.csv"), history)

    # loss curves
    plot_history(
        {"loss": history["loss"], "loss_phys": history["loss_phys"], "loss_data": history["loss_data"]},
        os.path.join(results_dir, "history.png"),
        title="Training history (inverse)",
    )

    # omega evolution plot
    import matplotlib.pyplot as plt  # local import to keep utils minimal

    plt.figure(figsize=(7, 3))
    plt.plot(history["omega"], label="PINN estimate")
    plt.hlines(omega_true, 0, len(history["omega"]), label="True value")
    plt.title("Omega estimation")
    plt.xlabel("Training step")
    plt.ylabel("ω")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "omega.png"), dpi=200)
    plt.close()

    # final prediction + csv
    u_pred = pinn(t_test).detach().cpu().numpy().reshape(-1)
    t_np = t_test.detach().cpu().numpy().reshape(-1)
    u_ex_np = u_exact.detach().cpu().numpy().reshape(-1)

    save_csv(
        os.path.join(results_dir, "prediction.csv"),
        header=["t", "u_exact", "u_pred"],
        rows=zip(t_np.tolist(), u_ex_np.tolist(), u_pred.tolist()),
    )

    save_csv(
        os.path.join(results_dir, "observations.csv"),
        header=["t_obs", "u_obs"],
        rows=zip(
            t_obs.detach().cpu().numpy().reshape(-1).tolist(),
            u_obs.detach().cpu().numpy().reshape(-1).tolist(),
        ),
    )

    plot_prediction(
        t=t_np,
        u_exact=u_ex_np,
        u_pred=u_pred,
        outpath=os.path.join(results_dir, "prediction.png"),
        title="Inverse PINN (final)",
        t_obs=t_obs.detach().cpu().numpy().reshape(-1),
        u_obs=u_obs.detach().cpu().numpy().reshape(-1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse PINN for SDOF oscillator (estimate omega).")
    parser.add_argument("--config", type=str, default="configs/inverse.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_inverse(cfg)


if __name__ == "__main__":
    main()
