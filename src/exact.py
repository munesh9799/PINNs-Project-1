from __future__ import annotations

import torch


def exact_solution(omega: float, xi: float, t: torch.Tensor, u0: float = 1.0, v0: float = 0.1) -> torch.Tensor:
    """
    Analytical solution for the underdamped SDOF oscillator:
        u¨ + 2*xi*omega*u˙ + omega^2*u = 0   (xi < 1)

    Important: implemented using torch operations (no numpy), to keep dtype/device consistent.
    """
    if not (xi < 1.0):
        raise ValueError("This exact solution assumes underdamped case: xi < 1.")

    omega_d = omega * torch.sqrt(torch.tensor(1.0 - xi**2, dtype=t.dtype, device=t.device))

    # u(t) = e^{-xi*omega*t} [ u0 cos(omega_d t) + ((v0 + xi*omega*u0)/omega_d) sin(omega_d t) ]
    A = u0
    B = (v0 + xi * omega * u0) / omega_d

    return torch.exp(-(xi * omega) * t) * (A * torch.cos(omega_d * t) + B * torch.sin(omega_d * t))
