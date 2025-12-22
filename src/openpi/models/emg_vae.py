# from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# import openpi.training.sharding as sharding

class EMGEncoder(nnx.Module):
    """Convolutional VAE Encoder for EMG."""

    def __init__(self, c_in: int, latent_dim: int, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.conv1 = nnx.Conv(c_in, 32, kernel_size=(5,), strides=(2,), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(5,), strides=(2,), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(64, 64, kernel_size=(3,), strides=(1,), padding="SAME", rngs=rngs)

        #mean and logvar layers
        self.mean = nnx.Conv(64, latent_dim, kernel_size=(1,), padding="SAME", rngs=rngs)
        self.logvar = nnx.Conv(64, latent_dim, kernel_size=(1,), padding="SAME", rngs=rngs)

    def __call__(self, x):  # x: [B, T, C]
        """Applies Encoder module."""
        # x = jnp.swapaxes(x, 1, 2) #[B, C, T]
        h = jax.nn.relu(self.conv1(x))
        h = jax.nn.relu(self.conv2(h))
        h = jax.nn.relu(self.conv3(h))

        mean = self.mean(h)
        logvar = self.logvar(h)

        #reparameterize
        noise = jax.random.normal(self.rngs.noise(), mean.shape)
        z = mean + jnp.exp(0.5*logvar)*noise

        #reshape
        # z = jnp.swapaxes(z, 1, 2)
        # mean = jnp.swapaxes(mean, 1, 2)
        # logvar = jnp.swapaxes(logvar, 1, 2)

        return z, mean, logvar
    
class EMGDecoder(nnx.Module):
    """Convolutional VAE Decoder for EMG."""

    def __init__(self, c_out: int, latent_dim: int, *, rngs: nnx.Rngs):
        self.rngs = rngs
        
        self.p1 = nnx.Conv(latent_dim, 64, kernel_size=(1,), padding="SAME", rngs=rngs)

        self.convT1 = nnx.ConvTranspose(64, 64, kernel_size=(5,), strides=(2,), padding="SAME", rngs=rngs)
        self.convT2 = nnx.ConvTranspose(64, 32, kernel_size=(5,), strides=(2,), padding="SAME", rngs=rngs)

        self.refine = nnx.Conv(32, 32, kernel_size=(3,), padding="SAME", rngs=rngs)
        self.out = nnx.Conv(32, c_out, kernel_size=(1,), padding="SAME", rngs=rngs)


    def __call__(self, z_tokens, T_target: int):  # z_tokens: [B, T', D].
        """Applies Decoder module."""
        # z = jnp.swapaxes(z_tokens, 1, 2) #[B, D, T']
        h = jax.nn.relu(self.p1(z_tokens))
        h = jax.nn.relu(self.convT1(h))
        h = jax.nn.relu(self.convT2(h))
        h = jax.nn.relu(self.refine(h))
        x_hat = self.out(h)

        T = x_hat.shape[1]
        if T == T_target:
            pass
        elif T > T_target:
            x_hat = x_hat[:, :T_target, :]
        elif T < T_target:
            x_hat = jnp.pad(x_hat, ((0, 0), (0, T_target - T), (0, 0)))
        
        # x_hat = jnp.swapaxes(x_hat, 1, 2)
        return x_hat

class EMGVAE(nnx.Module):
    def __init__(self, c_in, latent_dim, *, rngs: nnx.Rngs):
        self.encoder = EMGEncoder(c_in, latent_dim, rngs=rngs)
        self.decoder = EMGDecoder(c_in, latent_dim, rngs=rngs)
    
    def __call__(self, x):
        z_tokens, mean, logvar = self.encoder(x)
        x_hat = self.decoder(z_tokens, T_target=x.shape[1])
        return x_hat, mean, logvar, z_tokens