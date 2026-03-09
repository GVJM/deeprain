"""
thresholded_glow_mc.py — Glow with Threshold Transform for Zero-Inflated Precipitation

Uses the thresholding trick:
  Training:  y_target = where(x > 0, log1p(x), -1.0)  → flow computes exact NLL on y
  Sampling:  z ~ N(0,I) → inverse flow → expm1(relu(·))

Reuses ActNorm, InvertibleLinearLU from glow.py and _AffineCouplingCond from glow_mc.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS
from models.glow import ActNorm, InvertibleLinearLU
from models.glow_mc import _AffineCouplingCond


class ThresholdedGlowMc(BaseModel):
    """
    Glow with threshold transform for zero-inflated precipitation.

    Flow operates on the bimodal continuous distribution produced by the threshold
    transform, then sampling applies expm1(relu(·)) to recover precipitation amounts.
    """

    def __init__(
        self,
        input_size: int = 15,
        n_layers: int = 8,
        hidden_size: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size

        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        E = self.cond_block.total_dim

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(ActNorm(input_size))
            self.layers.append(InvertibleLinearLU(input_size))
            self.layers.append(_AffineCouplingCond(input_size, E, hidden_size))

        self._cond_probs: dict[str, np.ndarray] = {}

    def set_cond_distribution(self, cond_arrays: dict[str, np.ndarray]):
        self._cond_probs = {}
        for name, n_classes, _ in self.cond_block.categoricals:
            arr = cond_arrays[name].astype(int)
            counts = np.bincount(arr, minlength=n_classes).astype(float)
            self._cond_probs[name] = counts / counts.sum()
        self._continuous_data = {
            name: cond_arrays[name].astype(np.float32)
            for name, _ in self.cond_block.continuous
            if name in cond_arrays
        }

    def log_prob(self, y: Tensor, c_emb: Tensor) -> Tensor:
        z = y
        log_det_total = torch.zeros(y.shape[0], device=y.device)

        for layer in self.layers:
            if isinstance(layer, _AffineCouplingCond):
                z, log_det = layer(z, c_emb)
            else:
                z, log_det = layer(z)
            log_det_total += log_det

        log_pz = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi, device=y.device))).sum(dim=-1)
        return log_pz + log_det_total

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        # Threshold transform: wet → log1p(x), dry → -1.0
        y_target = torch.where(x > 0, torch.log1p(x), torch.tensor(-1.0, device=x.device))

        log_p = self.log_prob(y_target, c_emb)
        nll = -log_p.mean()
        return {'total': nll, 'nll': nll}

    def sample(self, n: int, cond: dict = None, steps=None, method=None) -> Tensor:
        device = next(self.parameters()).device

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        z = torch.randn(n, self.input_size, device=device)

        with torch.no_grad():
            for layer in reversed(self.layers):
                if isinstance(layer, _AffineCouplingCond):
                    z = layer.inverse(z, c_emb)
                else:
                    z = layer.inverse(z)

        # ReLU maps sentinel <= 0 → 0 (dry); expm1 reverts log1p
        return torch.expm1(F.relu(z))
