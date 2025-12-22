import numpy as np
import torch
from datetime import datetime
import os
from dataclasses import asdict
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax import serialization
import orbax.checkpoint as ocp
import json
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from openpi.models.emg_vae import EMGVAE
from openpi.models.emg_vae_config import EMGVAEConfig
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


# @nnx.jit 
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


def save_vae(model: nnx.Module, cfg: dict, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    graphdef, state = nnx.split(model)

    # Save graphdef/state using orbax PyTreeCheckpointer
    ckpt = {"graphdef": graphdef, "state": state}
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(out / "ckpt", ckpt, force=True)

    (out / "config.json").write_text(json.dumps(cfg, indent=2))

def load_vae(vae_ctor, in_dir: str, *, rngs: nnx.Rngs):
    inp = Path(in_dir)
    cfg = json.loads((inp / "config.json").read_text())

    # Rebuild template to get correct structure types
    template = vae_ctor(cfg, rngs)
    graphdef_t, state_t = nnx.split(template)

    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(inp / "ckpt", item={"graphdef": graphdef_t, "state": state_t})

    model = nnx.merge(ckpt["graphdef"], ckpt["state"])
    return model, cfg

if __name__ == "__main__":
    dataset = EMGIterableDataset(repo_id="jasontchan/task123-23-12345-12_MU", split="train", emg_key="emg", expected_shape=(50, 8), drop_bad_rows=False)
    loader = DataLoader(dataset, batch_size=32, 
                        num_workers=0,
                        collate_fn=emg_collate, 
                        pin_memory=True, 
                    )

    rngs = nnx.Rngs(0, noise=1)

    model = EMGVAE(c_in=8, latent_dim=64, rngs=rngs)
    cfg = EMGVAEConfig(
        c_in=8,
        latent_dim=64,
        widths=(32, 64, 64),
        kernels=(5, 5, 3),
        strides=(2, 2, 1),
    )
    lr = 1e-3
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    optimizer = nnx.Optimizer(model, opt)

    num_epochs = 25
    beta = 2.0 

    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_root = f"/lambda/nfs/filesystem1/openpi-vela/checkpoints/emg_vae/{timestamp_str}"
    os.makedirs(ckpt_root, exist_ok=True)

    best_loss = float("inf")
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
        # save per-epoch
        save_vae(model, asdict(cfg), f"{ckpt_root}/epoch_{epoch:03d}")

        # save best
        avg_loss = train_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_vae(model, asdict(cfg), f"{ckpt_root}/best")
        print(f"Train epoch: {epoch}, avg loss: {avg_loss:.5f}")