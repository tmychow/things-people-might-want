import jax.numpy as jnp
import torch
import torch.nn as tnn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from model import RMSNorm


BATCH_SIZE = 2
SEQ_LEN = 9
DIM_SIZE = 4096
EPS = 1e-05

@torch.no_grad
def test():
    rms_norm_pt = LlamaRMSNorm(hidden_size=DIM_SIZE, eps=EPS)
    rms_norm_pt.weight = tnn.Parameter(torch.randn_like(rms_norm_pt.weight))

    print(f'{rms_norm_pt.weight.shape=}')

    params_pt = rms_norm_pt.weight
    params_jax = jnp.asarray(params_pt.numpy())

    x_pt = torch.rand(BATCH_SIZE, SEQ_LEN, DIM_SIZE, dtype=torch.float)
    x_jax = jnp.asarray(x_pt.numpy()).astype(jnp.float32)

    assert x_pt.shape == x_jax.shape

    y_pt = rms_norm_pt(x_pt)
    y_jax = jnp.asarray(y_pt.numpy())

    rms_norm_jax = RMSNorm(None, "test", DIM_SIZE, EPS, jnp.float32)
    rms_norm_jax.w = params_jax

    y_hat_jax = rms_norm_jax(x_jax)
    #print(f'{y_jax[:30]=}')
    #print(f'{y_hat_jax[:30]=}')
    #assert jnp.allclose(y_jax, y_hat_jax, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(y_jax, y_hat_jax)


if __name__ == '__main__':
    test()