"""
glow.py — GLOW (Generative Flow with Invertible 1×1 Convolutions) para dados tabulares.

Arquitetura:
    n_steps blocos de: ActNorm → InvertibleLinearLU → AffineCouplingGLOW
    Loss: NLL exata = -log_prob(x)
    Sampling: z ~ N(0,I) → inverso das camadas em ordem reversa
"""

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import scipy.linalg

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


class InvertibleLinearLU(nn.Module):
    """
    Camada linear invertível parametrizada via decomposição LU.
    Calcula det(W) em O(D) como produto dos elementos diagonais de U.
    """
    def __init__(self, input_size: int):
        super().__init__()

        random_matrix = torch.randn(input_size, input_size)
        W_init, _ = torch.linalg.qr(random_matrix)

        np_W = W_init.numpy()
        P_init, L_init, U_init = scipy.linalg.lu(np_W)

        s_init = np.diag(U_init)
        sign_s_init = np.sign(s_init)
        log_s_init = np.log(np.abs(s_init))

        U_init = np.triu(U_init, k=1)
        L_init = np.tril(L_init, k=-1)

        self.register_buffer('P', torch.from_numpy(P_init).float())
        self.register_buffer('sign_s', torch.from_numpy(sign_s_init).float())

        self.L = nn.Parameter(torch.from_numpy(L_init).float())
        self.U = nn.Parameter(torch.from_numpy(U_init).float())
        self.log_s = nn.Parameter(torch.from_numpy(log_s_init).float())

        self.register_buffer('eye', torch.eye(input_size))

    def _assemble_W(self) -> Tensor:
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L @ U

    def forward(self, x: Tensor):
        W = self._assemble_W()
        y = x @ W
        log_det = self.log_s.sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, y: Tensor):
        W = self._assemble_W()
        W_inv = torch.linalg.inv(W)
        return y @ W_inv


class ActNorm(nn.Module):
    """Normalização de ativação com inicialização dependente dos dados."""
    def __init__(self, input_size: int):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, input_size))
        self.log_scale = nn.Parameter(torch.zeros(1, input_size))
        self.initialized = False

    def forward(self, x: Tensor):
        if not self.initialized:
            with torch.no_grad():
                self.loc.copy_(-x.mean(dim=0, keepdim=True))
                self.log_scale.copy_(-torch.log(x.std(dim=0, keepdim=True) + 1e-6))
                self.initialized = True

        y = x * torch.exp(self.log_scale) + self.loc
        log_det = self.log_scale.sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, y: Tensor):
        return (y - self.loc) * torch.exp(-self.log_scale)


class AffineCouplingGLOW(nn.Module):
    """Acoplamento afim GLOW: divide em metades, MLP(x1) → (s, t) para transformar x2."""
    def __init__(self, input_size: int, hidden: int = 128):
        super().__init__()
        self.n_half = input_size // 2
        n_out = input_size - self.n_half

        self.net = nn.Sequential(
            nn.Linear(self.n_half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out * 2),
        )

    def forward(self, x: Tensor):
        x1, x2 = x[:, :self.n_half], x[:, self.n_half:]
        st = self.net(x1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=-1)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: Tensor):
        y1, y2 = y[:, :self.n_half], y[:, self.n_half:]
        st = self.net(y1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=-1)


class GLOW(BaseModel):
    """
    Arquitetura GLOW para dados tabulares 1D.
    Cada bloco: ActNorm → InvertibleLinearLU → AffineCouplingGLOW
    """
    def __init__(self, input_size: int = 15, n_steps: int = 8, hidden_size: int = 128, **kwargs):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.ModuleList()
        for _ in range(n_steps):
            self.layers.append(ActNorm(input_size))
            self.layers.append(InvertibleLinearLU(input_size))
            self.layers.append(AffineCouplingGLOW(input_size, hidden_size))

    def log_prob(self, x: Tensor) -> Tensor:
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det_total

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """NLL exata. beta é ignorado (sem KL)."""
        log_p = self.log_prob(x)
        nll = -log_p.mean()
        return {'total': nll, 'nll': nll}

    def sample(self, n: int, steps=None, method=None) -> Tensor:
        device = next(self.parameters()).device
        z = torch.randn(n, self.input_size, device=device)
        with torch.no_grad():
            for layer in reversed(self.layers):
                z = layer.inverse(z)
        return z
