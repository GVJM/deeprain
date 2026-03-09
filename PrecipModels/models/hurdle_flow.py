"""
hurdle_flow.py — Conditional Normalizing Flow para Precipitação (Hurdle Flow)

Combina a modelagem exata de likelihood do RealNVP com a lógica de barreira (Hurdle)
para resolver o problema da massa de probabilidade no zero (dias secos).

Como funciona:
    1. Ocorrência (BCE): Uma MLP aprende a probabilidade de chuva por estação.
    2. Quantidade (NLL): Um RealNVP Condicional modela a quantidade de chuva.
       - A Flow é condicionada na máscara de ocorrência (onde choveu).
       - TRUQUE MATEMÁTICO: Para evitar que a Flow tente modelar o zero exato
         (o que explodiria o gradiente), substituímos os dias secos por ruído
         Gaussiano N(0,1) durante o treino. A Flow aprende a ignorar esse ruído
         graças ao condicionamento da máscara, focando apenas nos dias úmidos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel

class _OccurrenceMLP(nn.Module):
    """
    MLP idêntica à do hurdle_simple.py.
    Aprende um vetor fixo de probabilidades de ocorrência de chuva por estação.
    """
    def __init__(self, input_size: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Retorna probabilidades p_rain em [0, 1]"""
        return torch.sigmoid(self.net(x))


class _ConditionalCouplingLayer(nn.Module):
    """
    Camada de acoplamento afim condicionada (Conditional RealNVP).
    
    A grande diferença para o RealNVP padrão é que a MLP interna recebe 
    não apenas a metade do dado (x1), mas também a máscara de ocorrência (cond).
    Isso permite que a transformação saiba se a estação atual está chovendo ou não.
    """
    def __init__(self, input_size: int, mask: Tensor, hidden: int = 128):
        super().__init__()
        self.register_buffer('mask', mask.float())
        n_active = int(mask.sum().item())
        self.n_free = input_size - n_active

        # A entrada da MLP é: metade ativa do dado (n_active) + MÁSCARA COMPLETA (input_size)
        mlp_input_size = n_active + input_size
        
        self.net = nn.Sequential(
            nn.Linear(mlp_input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_free * 2),  # Saída: s e t para a metade livre
        )

    def forward(self, x: Tensor, cond: Tensor):
        """Passagem Data -> Latente (usada no treino)"""
        x1 = x[:, self.mask.bool()]   # Parte que não sofre alteração
        x2 = x[:, ~self.mask.bool()]  # Parte que será transformada

        # Concatena a parte fixa com a máscara de condição
        mlp_in = torch.cat([x1, cond], dim=1)
        st = self.net(mlp_in)
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)  # Limita 's' para evitar overflow no exponencial

        # Transformação afim
        y2 = x2 * torch.exp(s) + t
        
        cond_free = cond[:, ~self.mask.bool()]# Only dimensions that are "free" AND "raining" should contribute to log_det
        log_det = (s * cond_free).sum(dim=-1)

        y = torch.empty_like(x)
        y[:, self.mask.bool()] = x1
        y[:, ~self.mask.bool()] = y2
        return y, log_det

    def inverse(self, y: Tensor, cond: Tensor):
        """Passagem Latente -> Data (usada na geração)"""
        x1 = y[:, self.mask.bool()]
        y2 = y[:, ~self.mask.bool()]

        mlp_in = torch.cat([x1, cond], dim=1)
        st = self.net(mlp_in)
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)

        # Inverso da transformação afim
        x2 = (y2 - t) * torch.exp(-s)

        x = torch.empty_like(y)
        x[:, self.mask.bool()] = x1
        x[:, ~self.mask.bool()] = x2
        return x


class HurdleFlow(BaseModel):
    """
    Hurdle Flow: Modela a chuva separando a probabilidade de ocorrer (MLP)
    da distribuição de quantidade (Conditional RealNVP).
    """
    def __init__(
        self,
        input_size: int = 15,
        n_coupling: int = 8,
        hidden_size: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.occ_mlp = _OccurrenceMLP(input_size, hidden=32)
        
        # Cria as camadas de acoplamento com máscaras alternadas (checkerboard 1D)
        self.layers = nn.ModuleList()
        for i in range(n_coupling):
            mask = torch.zeros(input_size, dtype=torch.bool)
            if i % 2 == 0:
                mask[::2] = True
            else:
                mask[1::2] = True
            
            if mask.all() or (~mask).all():
                mask = torch.zeros(input_size, dtype=torch.bool)
                mask[:input_size // 2] = True
                
            self.layers.append(_ConditionalCouplingLayer(input_size, mask, hidden_size))

        self.log_scale = nn.Parameter(torch.zeros(input_size))
        self.max_log_scale = 3.0
        self.max_preexp = 10.0

    def _effective_log_scale(self) -> Tensor:
        """
        Limita a escala global da Flow para evitar explosões numéricas.
        Mantém gradientes suaves (tanh) e a transformação invertível.
        """
        return self.max_log_scale * torch.tanh(self.log_scale / self.max_log_scale)

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Loss = BCE (Ocorrência) + NLL (Flow nas quantidades)
        """
        occ_target = (x > 0).float()

        # 1. Loss de Ocorrência (BCE)
        dummy_input = torch.zeros_like(x)
        p_rain = self.occ_mlp(dummy_input)
        p_rain = p_rain.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(p_rain, occ_target, reduction='mean')

        # 2. Preparação do dado para a Flow
        # TRUQUE MATEMÁTICO: Onde choveu, usamos log1p(x) para ter um espaço contínuo.
        # Onde NÃO choveu, injetamos ruído N(0,1). Isso impede que a Flow colapse
        # tentando mapear o zero exato. Como a Flow é condicionada na máscara, 
        # ela aprende a ignorar esse ruído durante a geração.
        y_in = torch.where(
            occ_target > 0, 
            torch.log1p(x), 
            torch.randn_like(x)
        )

        # # 3. Passagem pela Flow (Data -> Latente)
        # z = y_in * torch.exp(self.log_scale)
        # log_det_total = self.log_scale.sum() * x.shape[0]

        # for layer in self.layers:
        #     z, log_det = layer(z, cond=occ_target)
        #     log_det_total += log_det

        # 3. Passagem pela Flow (Data -> Latente)
        eff_log_scale = self._effective_log_scale()
        z = y_in * torch.exp(eff_log_scale)
        
        # FIX: Initialize log_det_total with the batch size (B,) on the correct device
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        # Add the learned scale (automatic broadcast from scalar to batch vector)
        log_det_total += (eff_log_scale * occ_target).sum(dim=-1)
        
        for layer in self.layers:
            z, log_det = layer(z, cond=occ_target)
            log_det_total += log_det

        # log P(Z) sob uma Normal Padrão
        log_pz = (-0.5 * (z ** 2 + np.log(2 * np.pi)) * occ_target).sum(dim=-1)
        
        # NLL Total da Flow
        log_prob = log_pz + log_det_total
        nll = -log_prob.mean()

        total = bce + nll
        return {'total': total, 'bce': bce, 'nll': nll}

    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Gera amostras em duas etapas garantindo zeros perfeitos.
        """
        device = next(self.parameters()).device

        with torch.no_grad():
            # 1. Gera a máscara de ocorrência (chove ou não chove)
            dummy = torch.zeros(1, self.input_size, device=device)
            p_rain = self.occ_mlp(dummy).squeeze(0)
            occ_sim = torch.bernoulli(p_rain.expand(n, -1))

            # 2. Gera a quantidade via fluxo inverso (Latente -> Data)
            z = torch.randn(n, self.input_size, device=device)

            for layer in reversed(self.layers):
                z = layer.inverse(z, cond=occ_sim)

            eff_log_scale = self._effective_log_scale()
            z = z * torch.exp(-eff_log_scale)

            # Inverte o log1p para voltar à escala original
            # Clamp evita cauda explosiva (expm1 cresce muito rápido).
            a_raw = torch.expm1(z.clamp(min=-10.0, max=self.max_preexp))
            
            # Corta valores negativos gerados residualmente e aplica a máscara
            a_raw = F.relu(a_raw)
            precip = a_raw * occ_sim

        return precip
