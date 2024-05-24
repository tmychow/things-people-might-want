import torch
import torch.nn as tnn

import jax.numpy as jnp

from nn import Embedding

VOCAB_SIZE = 32000
DIM_SIZE = 4096
TOKEN_PAD_ID = 0


@torch.no_grad
def test():
    embedding_pt = tnn.Embedding(VOCAB_SIZE, DIM_SIZE, TOKEN_PAD_ID)
    embedding_pt.weight = tnn.Parameter(torch.randn_like(embedding_pt.weight))

    params_pt = embedding_pt.weight
    params_jax = jnp.asarray(params_pt.numpy())


    x_pt = torch.tensor([[12, 36, 5145, 21, 0, 556, 25454, 54, 563, 8]], dtype=torch.int)
    x_jax = jnp.asarray(x_pt.numpy()).astype(jnp.uint16)

    y_pt = embedding_pt(x_pt)
    y_jax = jnp.asarray(y_pt.numpy())

    embedding_jax = Embedding(VOCAB_SIZE, DIM_SIZE)
    embedding_jax.w = params_jax

    y_hat_jax = embedding_jax(x_jax)
    assert jnp.allclose(y_jax, y_hat_jax)
    #print(f'{y_jax[:30]=}')
    #print(f'{y_hat_jax[:30]=}')


if __name__ == '__main__':
    test()