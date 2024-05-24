
import jax.numpy as jnp
import torch
import torch.nn as tnn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

import pytest

from model import Attention
from model import ModelArgs
from model import precompute_freqs_cis

def convert_attention(attention_pt: LlamaAttention, config_pt: LlamaConfig) -> Attention:
    args = ModelArgs()
    assert args.dim == config_pt.hidden_size
    assert args.n_heads == config_pt.num_key_value_heads, (args.n_heads, config_pt.num_key_value_heads)
    attention_jax = Attention("test", args)
    assert attention_jax.n_rep == 1
    assert attention_jax.head_dim == attention_pt.head_dim
    attention_jax.wq.w = jnp.asarray(
        attention_pt.q_proj.weight.T.reshape(args.dim, attention_jax.n_rep, args.n_heads, attention_jax.head_dim).numpy()
    )
    attention_jax.wk.w = jnp.asarray(
        attention_pt.k_proj.weight.T.reshape(args.dim, args.n_heads, attention_jax.head_dim).numpy()
    )
    attention_jax.wv.w = jnp.asarray(
        attention_pt.v_proj.weight.T.reshape(args.dim, args.n_heads, attention_jax.head_dim).numpy()
    )
    attention_jax.wo.w = jnp.asarray(
        attention_pt.o_proj.weight.T.reshape(attention_jax.n_rep, args.n_heads, attention_jax.head_dim, args.dim).numpy()
    )
    return attention_jax


BATCH_SIZE = 1
SEQ_LEN = 7
DIM_SIZE = 4096
N_HEADS = 32
MAX_SEQ_LEN = 30
BEST_INTEGER = 3407
torch.manual_seed(BEST_INTEGER)

@torch.no_grad
@pytest.mark.skip(reason="TODO(pdex): fix this broken test")
def test():
    config_pt = LlamaConfig()
    attention_pt = LlamaAttention(config=config_pt)

    attention_jax = convert_attention(attention_pt, config_pt)

    x_pt = torch.rand(BATCH_SIZE, SEQ_LEN, DIM_SIZE, dtype=torch.float)
    x_jax = jnp.asarray(x_pt.numpy()).astype(jnp.float32)

    assert x_pt.shape == x_jax.shape

    mask_pt_1d = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)  # torch.rand(batch_size, seq_len) > 0.1
    mask_pt = torch.tril(torch.einsum('bi,bj->bij', mask_pt_1d, mask_pt_1d))[:, None]
    mask_jax_1d = jnp.asarray(mask_pt_1d.numpy())
    mask_jax = jnp.tril(jnp.einsum('bi,bj->bij', mask_jax_1d, mask_jax_1d))[:, None, None]

    # In the Hugging Face implementation, the attention mask is added to the attention
    # matrix, not multiplied.
    # See https://github.com/huggingface/transformers/issues/1935
    mask_pt = torch.where(mask_pt, 0, -10000.)

    y_pt = attention_pt(hidden_states=x_pt, attention_mask=mask_pt)[0]
    y_jax = jnp.asarray(y_pt.numpy())


    freqs_cis = precompute_freqs_cis(DIM_SIZE // N_HEADS, MAX_SEQ_LEN)
    start_pos = 0
    mask = jnp.full((SEQ_LEN, SEQ_LEN), float("-inf"))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((SEQ_LEN, start_pos)), mask], dtype=jnp.float32)
    y_hat_jax = attention_jax.forward(x_jax, start_pos, freqs_cis[start_pos:SEQ_LEN], mask)

    y_jax = jnp.where(mask_jax_1d[..., None], y_jax, 0.)
    y_hat_jax = jnp.where(mask_jax_1d[..., None], y_hat_jax, 0.)

    #print('y_jax', y_jax.reshape(-1)[:30])
    #print('y_hat_jax', y_hat_jax.reshape(-1)[:30])
    print(f'{y_jax.reshape(-1)[:30]=}')
    print(f'{y_hat_jax.reshape(-1)[:30]=}')
    tolerance=1e-03
    print(f'{tolerance=}')
    assert jnp.allclose(y_jax, y_hat_jax, rtol=tolerance, atol=tolerance)

if __name__ == '__main__':
    test()