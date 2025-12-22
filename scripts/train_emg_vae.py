import numpy as np
import torch

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from torch.utils.data import IterableDataset, DataLoader
from openpi.models.emg_vae import EMGVAE
from openpi.training.EMG_dataset import EMGIterableDataset, emg_collate

def torch_batch_to_jax(batch_torch: torch.Tensor) -> jax.Array:
    if batch_torch.is_cuda:
        batch_torch = batch_torch.detach().cpu()

    batch_np = batch_torch.detach().contiguous().numpy().astype(np.float32)
    return jnp.asarray(batch_np)

def vae_loss(x: jax.Array, x_hat: jax.Array, mu: jax.Array, logvar: jax.Array,
             beta: float = 1.0) -> jax.Array:

    recon = jnp.mean((x_hat - x) ** 2)

    kl_per = 0.5 * (jnp.exp(logvar) + (mu ** 2) - 1.0 - logvar)
    kl = jnp.mean(jnp.sum(kl_per, axis=-1))

    return recon + beta * kl


@nnx.jit 
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: jax.Array,
               beta: float) -> jax.Array:
    """
    model: EMGVAE nnx.Module
    optimizer: nnx.Optimizer wrapping optax
    batch: (B, T, C) jax.Array
    """
    def loss_fn(m: nnx.Module):
        out = m(batch)
        # Support both (x_hat, mu, logvar) and (x_hat, mu, logvar, z_tokens)
        x_hat, mu, logvar = out[:3]
        return vae_loss(batch, x_hat, mu, logvar, beta=beta)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

dataset = EMGIterableDataset(repo_id="jasontchan/task123-23-12345-12_MU", split="train", emg_key="emg", expected_shape=(100, 8), drop_bad_rows=True)
loader = DataLoader(dataset, batch_size=32, 
                     num_workers=0,
                     collate_fn=emg_collate, 
                     pin_memory=True, 
                   )

rngs = nnx.Rngs(0, noise=1)

model = EMGVAE(c_in=8, latent_dim=128, rngs=rngs)

lr = 1e-3
opt = optax.adam(lr)
optimizer = nnx.Optimizer(model, opt)

num_epochs = 10
beta = 2.0 

for epoch in range(num_epochs):
    train_loss = 0.0
    n_batches = 0

    for i, batch_torch in enumerate(loader):
        batch = torch_batch_to_jax(batch_torch)

        loss = train_step(model, optimizer, batch, beta=beta)
        train_loss += float(loss)
        n_batches += 1

        if (i + 1) % 50 == 0:
            print(f"Epoch {epoch} | step {i+1} | loss {float(loss):.5f}")

    print(f"Train epoch: {epoch}, avg loss: {train_loss / max(n_batches, 1):.5f}")
