from dataclasses import asdict, dataclass
from flax import nnx

from openpi.models.emg_vae import EMGVAE

@dataclass
class EMGVAEConfig:
    c_in: int = 8
    latent_dim: int = 64
    widths: tuple = (32, 64, 64)
    kernels: tuple = (5, 5, 3)
    strides: tuple = (2, 2, 1)

    def to_dict(self): return asdict(self)


def make_vae_from_config(cfg_dict, rngs: nnx.Rngs):
    cfg = EMGVAEConfig(**cfg_dict)
    return EMGVAE(
        c_in=cfg.c_in,
        latent_dim=cfg.latent_dim,
        rngs=rngs
    )