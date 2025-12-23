from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from .exact import exact_solution
from .models import FCN
from .utils import ensure_dir, load_config, plot_history, plot_prediction, save_csv, save_history_csv, set_seed


def train_forward_scaled(cfg: Dict) -> None:
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

    omega = float(cfg["sdof"]["omega"])
    xi = float(cfg["sdof"]["xi"])
    u0 = float(cfg["sdof"]["u0"])
    v0 = float(cfg["sdof"]["v0"])

    a_init = float(cfg["ansatz"]["a_init"])
    b_init = float(cfg["ansatz"]["b_init"])

    steps = int(cfg["train"]["steps"])
    lr = float(cfg["train"]["lr"])
    lambda1 = float(cfg["train"]["lambda1"])
    lambda2 = float(cfg["train"]["lambda2"])
    plot_every = int(cfg["train"]["plot_every"])

    # -------------------------
    # model + ansatz parameters (workshop-equivalent)
    # -------------------------
    pinn = FCN(1, 1, n_hidden, n_layers).to(device)

    a = torch.nn.Parameter(torch.tensor([a_init], dtype=torch.float32, device=device, requires_grad=True))
    b = torch.nn.Parameter(torch.tensor([b_init], dtype=torch.float32, device=device, requires_grad=True))

    t_boundary = torch.tensor(t0, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    t_physics = torch.linspace(t0, t1, n_physics, device=device).view(-1, 1).requires_grad_(True)

    t_test = torch.linspace(t0, t1, n_test, device=device).view(-1, 1)
    u_exact = exact_solution(omega, xi, t_test, u0=u0, v0=v0)

    mu = 2.0 * xi * omega
    k = omega**2

    optimiser = torch.optim.Adam(list(pinn.parameters()) + [a, b], lr=lr)

    history: Dict[str, List[float]] = {"loss": [], "loss_u0": [], "loss_v0": [], "loss_phys": [], "a": [], "b": []}

    # -------------------------
    # training
    # -------------------------
    for i in range(steps):
        optimiser.zero_grad()

        # boundary loss using ansatz: u = NN(t)*sin(a t + b)
        u_b = pinn(t_boundary) * torch.sin(a * t_boundary + b)
        loss_u0 = (torch.squeeze(u_b) - u0) ** 2

        dudt_b = torch.autograd.grad(u_b, t_boundary, torch.ones_like(u_b), create_graph=True)[0]
        loss_v0 = (torch.squeeze(dudt_b) - v0) ** 2

        # physics loss
        u_p = pinn(t_physics) * torch.sin(a * t_physics + b)
        dudt_p = torch.autograd.grad(u_p, t_physics, torch.ones_like(u_p), create_graph=True)[0]
        d2udt2_p = torch.autograd.grad(dudt_p, t_physics, torch.ones_like(dudt_p), create_graph=True)[0]
        loss_phys = torch.mean((d2udt2_p + mu * dudt_p + k * u_p) ** 2)

        loss = loss_u0 + lambda1 * loss_v0 + lambda2 * loss_phys
        loss.backward()
        optimiser.step()

        history["loss"].append(float(loss.detach().cpu()))
        history["loss_u0"].append(float(loss_u0.detach().cpu()))
        history["loss_v0"].append(float(loss_v0.detach().cpu()))
        history["loss_phys"].append(float(loss_phys.detach().cpu()))
        history["a"].append(float(a.detach().cpu()))
        history["b"].append(float(b.detach().cpu()))

        if i % plot_every == 0:
            u_pred = (pinn(t_test) * torch.sin(a * t_test + b)).detach().cpu().numpy().reshape(-1)
            t_np = t_test.detach().cpu().numpy().reshape(-1)
            u_ex_np = u_exact.detach().cpu().numpy().reshape(-1)
            plot_prediction(
                t=t_np,
                u_exact=u_ex_np,
                u_pred=u_pred,
                outpath=os.path.join(results_dir, f"pred_step_{i}.png"),
                title=f"Forward (scaled) PINN (step {i})",
            )

    # -------------------------
    # save artifacts
    # -------------------------
    save_history_csv(os.path.join(results_dir, "history.csv"), history)
    plot_history(
        {"loss": history["loss"], "loss_u0": history["loss_u0"], "loss_v0": history["loss_v0"], "loss_phys": history["loss_phys"]},
        os.path.join(results_dir, "history.png"),
        title="Training history (forward_scaled)",
    )

    # final prediction + csv
    u_pred = (pinn(t_test) * torch.sin(a * t_test + b)).detach().cpu().numpy().reshape(-1)
    t_np = t_test.detach().cpu().numpy().reshape(-1)
    u_ex_np = u_exact.detach().cpu().numpy().reshape(-1)

    save_csv(
        os.path.join(results_dir, "prediction.csv"),
        header=["t", "u_exact", "u_pred"],
        rows=zip(t_np.tolist(), u_ex_np.tolist(), u_pred.tolist()),
    )

    plot_prediction(
        t=t_np,
        u_exact=u_ex_np,
        u_pred=u_pred,
        outpath=os.path.join(results_dir, "prediction.png"),
        title="Forward (scaled) PINN (final)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward (scaled ansatz) PINN for SDOF oscillator.")
    parser.add_argument("--config", type=str, default="configs/forward_scaled.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_forward_scaled(cfg)


if __name__ == "__main__":
    main()
