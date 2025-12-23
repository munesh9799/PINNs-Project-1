from __future__ import annotations

import torch
import torch.nn as nn


class FCN(nn.Module):
    """
    Standard fully-connected network (workshop-equivalent architecture).

    - Input: (N, N_INPUT)
    - Output: (N, N_OUTPUT)
    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_layers: int) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        activation = nn.Tanh

        self.fcs = nn.Sequential(nn.Linear(n_input, n_hidden), activation())

        hidden_blocks = []
        for _ in range(n_layers - 1):
            hidden_blocks.append(nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()))
        self.fch = nn.Sequential(*hidden_blocks)

        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fcs(x)
        x = self.fch(x)
        return self.fce(x)
