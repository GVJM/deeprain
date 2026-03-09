import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


class _MiniVAE(nn.Module):
    """Sub-VAE genérico com encoder 3-camadas e decoder 3-camadas."""

    def __init__(self, input_size: int, latent_size: int, output_activation: str = "none",
                 output_size: int = None):
        super().__init__()
        out_size = output_size if output_size is not None else input_size
        h = max(latent_size * 2, input_size)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, h),
            nn.LeakyReLU(),
            nn.Linear(h, h),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(h, latent_size)
        self.fc_logvar = nn.Linear(h, latent_size)

        dec_layers = [
            nn.Linear(latent_size, h),
            nn.LeakyReLU(),
            nn.Linear(h, h),
            nn.LeakyReLU(),
            nn.Linear(h, out_size),
        ]
        if output_activation == "sigmoid":
            dec_layers.append(nn.Sigmoid())
        elif output_activation == "relu":
            dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


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
            
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, h_dec), nn.LeakyReLU(),
            nn.Linear(h_dec, h_dec), nn.LeakyReLU(),
            nn.Linear(h_dec, input_size*2)
        )

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
        zc = torch.cat([z, c], dim=1)
        out =  self.decoder(zc)
        mu = out[:, :self.input_size]
        sigma = F.softplus(out[:, self.input_size:]) + 1e-4
        return mu, sigma

    def forward(self, x: Tensor, c: Tensor):
        mu_z, logvar_z = self.encode(x, c)
        z = self.reparameterize(mu_z, logvar_z)
        mu_out, sigma_out = self.decode(z, c)
        return mu_out, sigma_out, mu_z, logvar_z

class HurdleVAECondNll(BaseModel):
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

        # OccurrenceVAE: entrada binária (S,) → saída probabilidades (S,)
        self.occ_vae = _MiniVAE(input_size, latent_occ, output_activation="none")

        # AmountVAE: entrada [amount | occ_mask] (2S,) → saída quantidades (S,)
        self.amt_vae = _ConditionalMiniVAE(
            input_size=input_size,
            cond_size=input_size,
            latent_size=latent_amt,
            output_activation="relu"
        )

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Loss combinada: BCE_occ + beta*KL_occ + MSE_wet + beta*KL_amt

        Args:
            x: (B, S) — dados normalizados
        """

        # --- Ocorrência ---
        occ_target = (x > 0).float()
        p_hat, mu_o, logvar_o = self.occ_vae(occ_target)
        bce = F.binary_cross_entropy_with_logits(p_hat, occ_target, reduction='mean')
        kl_occ = -0.5 * torch.mean(1 + logvar_o - mu_o.pow(2) - logvar_o.exp())

        # --- Quantidade (condicionado na ocorrência) ---
        x_masked = x * occ_target  # zera os dias secos
        # log1p comprime a distribuição e estabiliza o KL (precipitação é log-normal)
        x_log = torch.log1p(x_masked)

        mu_hat, sigma_hat, mu_a, logvar_a = self.amt_vae(x_log, occ_target)
        kl_amt = -0.5 * torch.mean(1 + logvar_a - mu_a.pow(2) - logvar_a.exp())

        # NLL log-normal no espaço log1p apenas nas estações úmidas
        wet_mask = occ_target.bool()
        if wet_mask.any():
            log_x = torch.log1p(x[wet_mask])  # alvo no espaço log1p
            nll_wet = 0.5 * torch.mean(
                ((log_x - mu_hat[wet_mask]) / sigma_hat[wet_mask]) ** 2
                + 2 * torch.log(sigma_hat[wet_mask])
                + log_x  # Jacobiano: d(log1p(x))/dx = 1/(1+x) → -log(1/(1+x)) = log1p(x)
            )
        else:
            nll_wet = torch.tensor(0.0, device=x.device)

        total = bce + nll_wet + beta * (kl_occ + kl_amt)
        return {
            'total': total,
            'bce': bce,
            'nll_wet': nll_wet,
            'kl_occ': kl_occ,
            'kl_amt': kl_amt,
        }

    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Geração em dois passos:
            1. Ocorrência: z_o ~ N(0,I) → OccurrenceVAE → Bernoulli
            2. Quantidade: z_a ~ N(0,I) → AmountVAE([zeros, occ]) → ReLU
        """
        device = next(self.parameters()).device

        with torch.no_grad():
            # Passo 1: ocorrência
            z_o = torch.randn(n, self.latent_occ, device=device)
            p_rain = torch.sigmoid(self.occ_vae.decode(z_o))  # (n, S) — probabilidades
            occ_sim = torch.bernoulli(p_rain)   # (n, S) — binário

            # Passo 2: quantidade
            z_a = torch.randn(n, self.latent_amt, device=device)
            mu, sigma = self.amt_vae.decode(z_a, occ_sim)    # (n, S)

            noise = torch.randn_like(mu)
            log_a = mu + sigma * noise  # amostras no espaço log1p
            a_raw = torch.expm1(log_a.clamp(max=20))  # inverte log1p → escala original

            precip = a_raw * occ_sim
        return precip
