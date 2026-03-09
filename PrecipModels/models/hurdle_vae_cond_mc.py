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


class _ConditionalMiniVAE(nn.Module):
    """CVAE: Condiciona a geração a uma variável externa (ex: máscara de ocorrência)."""

    def __init__(self, input_size: int, cond_size: int, latent_size: int, output_activation: str = "none"):
        super().__init__()
        self.input_size = input_size
        self.cond_size = cond_size

        # ENCODER: Input becomes (input_size + cond_size)
        enc_in = input_size + cond_size
        h_enc = max(latent_size * 2, enc_in)
        print(f'COND VAE: enc_in = {enc_in} | h_enc = {h_enc}')
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, h_enc),
            nn.LeakyReLU(),
            nn.Linear(h_enc, h_enc),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(h_enc, latent_size)
        self.fc_logvar = nn.Linear(h_enc, latent_size)

        # DECODER: Input becomes (latent_size + cond_size)
        dec_in = latent_size + cond_size
        h_dec = max(latent_size * 2, dec_in)
        dec_layers = [
            nn.Linear(dec_in, h_dec),
            nn.LeakyReLU(),
            nn.Linear(h_dec, h_dec),
            nn.LeakyReLU(),
            nn.Linear(h_dec, input_size), # Output is just the rain amounts
        ]
        
        if output_activation == "relu":
            dec_layers.append(nn.ReLU())
            
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor, c: Tensor):
        # Concatena a quantidade de chuva (x) com a máscara de onde choveu (c)
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, c: Tensor) -> Tensor:
        # Concatena o ruído latente (z) com a máscara de onde choveu (c)
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)

    def forward(self, x: Tensor, c: Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

class HurdleVAECondMc(BaseModel):
    """
    Dois VAEs especializados: um para ocorrência binária, um para quantidade.
    """

    def __init__(
        self,
        input_size: int = 15,
        latent_occ: int = 32,
        latent_amt: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_occ = latent_occ
        self.latent_amt = latent_amt


        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
          # 6 para month-only

        # OccurrenceVAE: entrada binária (S,) → saída logits (S,)
        self.occ_vae = _ConditionalMiniVAE(
            input_size=input_size,
            cond_size=self.cond_block.total_dim,
            latent_size=latent_occ,
            output_activation="none"
        )


        # AmountVAE: entrada [amount | occ_mask] (2S,) → saída quantidades (S,)
        self.amt_vae = _ConditionalMiniVAE(
            input_size=input_size,
            cond_size=input_size + self.cond_block.total_dim,
            latent_size=latent_amt,
            output_activation="relu"
        )
        # Distribuição empírica dos condicionadores
        self._cond_probs: dict[str, np.ndarray] = {}

    def set_cond_distribution(self, cond_arrays: dict[str, np.ndarray]):
        """
        Armazena probabilidades empíricas para uso em sample() sem cond explícito.

        Args:
            cond_arrays: dict[str, ndarray(N,)] — arrays de condicionamento do treino
        """
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

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        """
        Loss combinada: BCE_occ + beta*KL_occ + MSE_wet + beta*KL_amt

        Args:
            x: (B, S) — dados normalizados
        """
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        # --- Ocorrência ---
        occ_target :Tensor = (x > 0).float()
        p_hat, mu_o, logvar_o = self.occ_vae(occ_target, c_emb)
        bce = F.binary_cross_entropy_with_logits(p_hat, occ_target, reduction='mean')
        kl_occ = -0.5 * torch.mean(1 + logvar_o - mu_o.pow(2) - logvar_o.exp())
        
        # --- Quantidade (condicionado na ocorrência) ---
        x_masked = x * occ_target  # zera os dias secos
        # log1p comprime a distribuição (precipitação é log-normal) e estabiliza o KL
        x_log = torch.log1p(x_masked)
        x_hat_log, mu_a, logvar_a = self.amt_vae(x_log, torch.cat([occ_target, c_emb], dim=-1))
        kl_amt = -0.5 * torch.mean(1 + logvar_a - mu_a.pow(2) - logvar_a.exp())

        # MSE no espaço log1p apenas nas estações úmidas
        wet_mask = occ_target.bool()
        if wet_mask.any():
            mse_wet = F.mse_loss(x_hat_log[wet_mask], x_log[wet_mask], reduction='mean')
        else:
            mse_wet = torch.tensor(0.0, device=x.device)

        total = 40 * bce + mse_wet +  beta * (100 * kl_occ + kl_amt)
        return {
            'total': total,
            'bce': bce,
            'mse_wet': mse_wet,
            'kl_occ': kl_occ,
            'kl_amt': kl_amt,
        }

    def sample(self, n: int, cond: dict = None, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Geração em dois passos:
            1. Ocorrência: z_o ~ N(0,I) → OccurrenceVAE → Bernoulli
            2. Quantidade: z_a ~ N(0,I) → AmountVAE([zeros, occ]) → ReLU
        """
        device = next(self.parameters()).device

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)


        with torch.no_grad():
            # Passo 1: ocorrência (decoder emite logits → sigmoid → Bernoulli)
            z_o = torch.randn(n, self.latent_occ, device=device)
            p_rain = torch.sigmoid(self.occ_vae.decode(z_o, c_emb))  # (n, S) — probabilidades
            occ_sim = torch.bernoulli(p_rain)                  # (n, S) — binário

            # Passo 2: quantidade no espaço log1p → expm1 para escala original
            z_a = torch.randn(n, self.latent_amt, device=device)
            a_log = F.relu(self.amt_vae.decode(z_a, torch.cat([occ_sim, c_emb], dim=-1)))  # não-negativo em log1p
            a_raw = torch.expm1(a_log)                          # inverte log1p

            # Combina: apenas estações com ocorrência recebem quantidade
            precip = a_raw * occ_sim
        return precip
