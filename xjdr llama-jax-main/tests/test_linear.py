import torch
import torch.nn as tnn

import jax.numpy as jnp

from nn import Linear

VOCAB_SIZE = 32000
DIM_SIZE = 4096
HIDDEN_SIZE = 11008
TOKEN_PAD_ID = 0


from typing import NamedTuple

class LinearSize(NamedTuple):
    name: str
    in_features: int
    out_features: int
    tolerance: float

sizes = [
    LinearSize(name="regular degular", in_features=DIM_SIZE, out_features=DIM_SIZE, tolerance=1e-04),
    LinearSize(name="hidden left", in_features=HIDDEN_SIZE, out_features=DIM_SIZE, tolerance=1e-03),
    LinearSize(name="hidden right", in_features=DIM_SIZE, out_features=HIDDEN_SIZE, tolerance=1e-03),
    LinearSize(name="output", in_features=DIM_SIZE, out_features=VOCAB_SIZE, tolerance=1e-03),
]

@torch.no_grad
def linear_test(name, in_features, out_features, tolerance):
    linear_pt = tnn.Linear(in_features, out_features, bias=False)
    linear_pt.weight = tnn.Parameter(torch.randn_like(linear_pt.weight))

    print(f'{linear_pt.weight.shape=}')

    params_pt = linear_pt.weight
    params_jax = jnp.asarray(params_pt.numpy())



    x_pt = torch.randn(in_features, dtype=torch.float).view(1, 1, in_features)
    x_jax = jnp.asarray(x_pt.numpy()).astype(jnp.float32)

    assert x_pt.shape == x_jax.shape, name

    y_pt = linear_pt(x_pt)
    y_jax = jnp.asarray(y_pt.numpy())

    linear_jax = Linear(None, "test", in_features, out_features)
    linear_jax.w = params_jax

    y_hat_jax = linear_jax(x_jax)
    print(f'{name} {y_jax[:30]=}')
    print(f'{name} {y_hat_jax[:30]=}')
    assert jnp.allclose(y_jax, y_hat_jax, rtol=tolerance, atol=tolerance), name


def test():
    for size in sizes:
        linear_test(size.name, size.in_features, size.out_features, size.tolerance)