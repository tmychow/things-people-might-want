from typing import Dict, List, Tuple
from typing import Optional

from pathlib import Path

import jax
from jax import numpy as jnp

from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PositionalSharding as PS
from jax.experimental import mesh_utils


class Module:
    """Base class for PyTorch-like modules."""

    @classmethod
    def add_prefix(cls, prefix: Optional[str], name: str) -> str:
        if prefix:
            return f'{prefix}.{name}'
        return name

    def __init__(self, name: str = None, needs_weight: bool = False, debug: bool = False):
        self.w: jnp.ndarray = None
        self.name: str = name
        self.debug: bool = debug
        self.needs_weight: bool = needs_weight
        if self.debug:
            print(self)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Forward pass not implemented for {type(self)}.")

    def get_submodules(self) -> Tuple["Module"]:
        return tuple()

    def has_submodules(self) -> bool:
        return len(self.get_submodules()) > 0

    def load_weights(self, path: Path, debug: bool, is_tpu: bool = False):
        if self.needs_weight:
            file_name = f'{self.name}.weight.npy'
            file = path / file_name
            assert file.exists() and file.is_file(), file
            if debug:
                print(f'Loading weights for {self.name} from {file_name}')
            self.w = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
            if is_tpu:
                self.shard()
        else:
            print(f'load_weights called on {self.name} when needs_weight is False')

    def load_state_dict(self, ckpt_path: Path, debug: bool = True, is_tpu: bool = False):
        for module in self.get_submodules():
            if module.has_submodules():
                module.load_state_dict(ckpt_path, debug, is_tpu)
            else:
                module.load_weights(ckpt_path, debug, is_tpu)

    def shard(self):
        """Shard the weights. Don't shard the norms"""
        if self.w is None:
            print("Load the weight first dummy")
            return
        # This should be configurable
        devices = mesh_utils.create_device_mesh((4, 2))
        mp = 'mp'
        fsdp = 'fsdp'
        mesh = Mesh(devices, axis_names=(mp, fsdp))
        if 'norm' in self.name:
            return
        elif 'tok_embeddings' in self.name:
            self.w = jax.device_put(self.w, NamedSharding(mesh, PS(fsdp, mp))) # Row Parallel
        elif 'w2' in self.name:
            self.w = jax.device_put(self.w, NamedSharding(mesh, PS(fsdp, mp))) # Row Parallel
        else:
            self.w = jax.device_put(self.w, NamedSharding(mesh, PS(mp, fsdp))) # Col Parallel
        if self.debug:
            jax.debug.visualize_array_sharding(self.w)


class ModuleList(Module):
    def __init__(self):
        super().__init__('layers')
        self.l: List[Module] = []

    def append(self, layer: Module):
        self.l.append(layer)

    def at(self, idx: int):
        return self.l[idx]

    def __len__(self):
        return len(self.l)

    def __iter__(self):
        return iter(self.l)

    def get_submodules(self) -> Tuple[Module]:
        return tuple(self.l)


# test_embedding.py
class Embedding(Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__(name='tok_embeddings', needs_weight=True)
        self.vocab_size: int = vocab_size
        self.dim: int = dim

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        return self.w[tokens]


# test_rmsnorm.py
class RMSNorm(Module):
    def __init__(self, prefix: Optional[str], name: str, dim: int, eps: float = 1e-5, dtype: jnp.dtype = jnp.bfloat16):
        super().__init__(name=nn.Module.add_prefix(prefix, name), needs_weight=True)
        self.dim: int = dim
        self.eps: float = eps
        self.dtype: jnp.dtype = dtype

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        output = self._norm(x).astype(self.dtype)
        return output * self.w


# test_linear.py
class Linear(Module):
    def __init__(self, prefix: Optional[str], name: str, n: int, d: int, bias: bool=False):
        super().__init__(name=Module.add_prefix(prefix, name), needs_weight=True)
        self.precision: jax.lax.Precision = jax.lax.Precision("default")
        self.n = n
        self.d = d

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # print(f'Linear({self.name}).__call__: {x.shape=}, {self.w.shape=}')
        if len(x.shape) == 2:
            y = jax.lax.dot_general(
              x,
              self.w.swapaxes(1, 0).reshape(self.n, self.d),
              (((x.ndim - 1,), (0,)), ((), ())),
                precision=self.precision,
            )
        elif len(x.shape) == 3:
            y = jax.lax.dot_general(
              x,
              self.w.swapaxes(1, 0).reshape(1, self.n, self.d),
              (((x.ndim - 1,), (1,)), ((0,), (0,))),
                precision=self.precision,
            )
        else:
            print("MA'AM THIS IS A WENDY'S DRIVETHROUGH!")
            y = None

        return y