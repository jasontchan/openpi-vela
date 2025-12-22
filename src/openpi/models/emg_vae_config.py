from dataclasses import asdict, dataclass

@dataclass
class EMGVAEConfig:
    c_in: int = 8
    latent_dim: int = 64
    widths: tuple = (32, 64, 64)
    kernels: tuple = (5, 5, 3)
    strides: tuple = (2, 2, 1)

    def to_dict(self): return asdict(self)