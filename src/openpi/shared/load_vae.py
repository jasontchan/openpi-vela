import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
import json
from pathlib import Path

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