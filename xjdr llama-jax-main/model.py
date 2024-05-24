from typing import Optional, Tuple, List

from dataclasses import dataclass

import jax
import jax.numpy as jnp

import nn


@dataclass
class MoeArgs:
    num_experts: int
    num_experts_per_tok: int


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5  # test the differnce beterrn 1e-5 and 1e-6
    sliding_window: int = 4096
    head_dim: int = 128
    hidden_dim: int = -1
    rope_theta: int = -1
    max_batch_size: int = 32
    max_seq_len: int = 2048
    moe: Optional[MoeArgs] = None


# test_rmsnorm.py
class RMSNorm(nn.Module):
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


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 1000000.0, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end)  # type: ignore
    freqs = jnp.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    freqs_cis = jnp.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def reshape_for_broadcast(freqs_cis: jnp.ndarray, x: jnp.ndarray):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(*shape)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(dtype), xk_out.astype(dtype)


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    batch_size, seq_len, num_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    expanded_kv = jnp.tile(x, n_rep)

    # Reshape to combine repeated key-value pairs
    reshaped_kv = expanded_kv.reshape(batch_size, seq_len, num_heads * n_rep, head_dim)
    return reshaped_kv


class Attention(nn.Module):
    def __init__(self, prefix: str, args: ModelArgs):
        super().__init__(name=nn.Module.add_prefix(prefix, 'attention')),
        self.args = args
        self.n_kv_heads: int = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(self.name, 'wq', args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.name, 'wk', args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.name, 'wv', args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.name, 'wo', args.n_heads * self.head_dim, args.dim, bias=False)
        self.cache_k = jnp.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = jnp.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def get_submodules(self) -> Tuple[nn.Module]:
        return (self.wq, self.wk, self.wv, self.wo)

    def forward(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        # Get shape information
        bsz, seqlen, _ = x.shape
        x = x.astype(jnp.bfloat16)
        # Project input to query, key, and value heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to standard format
        xq, xk, xv = (
            xq.reshape(bsz, seqlen, self.n_local_heads, -1),
            xk.reshape(bsz, seqlen, self.n_local_kv_heads, -1),
            xv.reshape(bsz, seqlen, self.n_local_kv_heads, -1),
        )

        # Apply rotary positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Update cache with current key and value
        self.cache_k = self.cache_k.at[:bsz, start_pos : start_pos + seqlen].set(xk)
        self.cache_v = self.cache_v.at[:bsz, start_pos : start_pos + seqlen].set(xv)

        # Extract attention keys and values from cache
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)      # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # Transpose query, keys, and values
        xq = jnp.swapaxes(xq, 1, 2)          # (bs, n_local_heads, seqlen, head_dim)
        keys = jnp.swapaxes(keys, 1, 2)      # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = jnp.swapaxes(values, 1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        # Attention scoring with scaling factor incorporated into einsum
        scores = jnp.einsum("bhsd,bhdl->bhsl", xq, keys.swapaxes(2,3)) / jnp.sqrt(self.head_dim)  # (bs, n_local_heads, seqlen, cache_len + seqlen)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax and attention output
        scores = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum("bhsl,bhld->bhsd", scores, values)  # (bs, n_local_heads, seqlen, head_dim)

        # Transform output and apply final linear projection
        output = jnp.swapaxes(output, 1, 2).reshape(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        prefix: str,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mistral: bool,
        moe: bool = False,
        expert: int = 0
    ):
        if moe:
            super().__init__(name=nn.Module.add_prefix(prefix, f"feed_forward.experts.{expert}")),
        else:
            super().__init__(name=nn.Module.add_prefix(prefix, "feed_forward")),
        if not mistral:
            hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(self.name, 'w1', dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(self.name, 'w2', hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(self.name, 'w3', dim, hidden_dim, bias=False)

    def get_submodules(self) -> Tuple[nn.Module]:
        return (self.w1, self.w2, self.w3)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class MoeLayer(nn.Module):
    def __init__(self, prefix: str, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__(name=nn.Module.add_prefix(prefix, "feed_forward.experts"), needs_weight=True),
        assert len(experts) > 0
        self.experts = nn.ModuleList()
        self.gate = gate
        self.args = moe_args
        for expert in experts:
            self.experts.append(expert)

    def forward(self, xb: jnp.ndarray) -> jnp.ndarray:
        # TODO(xjdr): vmap / pmap / xmap over batch dimension in the future
        x = xb[0]
        gate_logits = self.gate(x)
        weights, selected_experts = jax.lax.top_k(gate_logits, 2)
        weights = jax.nn.softmax(weights, axis=1).astype(jnp.float32)
        results = jnp.zeros_like(x)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = jnp.where(selected_experts == i)
            # TODO(xjdr): Add this back with quantization work
            #with jax.numpy_dtype_promotion('strict'):
            results = results.at[batch_idx].set(results[batch_idx] + weights[batch_idx, nth_expert, None] * expert.forward(x[batch_idx]))
        return results

    def get_submodules(self) -> Tuple[nn.Module]:
        return (self.experts, self.gate)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, moe: bool = False):
        super().__init__(name=f'layers.{layer_id}')
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(self.name, args)
        if moe:
            self.feed_forward = MoeLayer(
                self.name,
                experts=[FeedForward(
                    self.name,
                    args.dim,
                    hidden_dim=4 * args.dim if args.hidden_dim == -1 else args.hidden_dim,
                    multiple_of=args.multiple_of, 
                    ffn_dim_multiplier=args.ffn_dim_multiplier, 
                    mistral=True,
                    moe = True,
                    expert = i
                ) for i in range(args.moe['num_experts'])],
                gate=nn.Linear(self.name, 'feed_forward.gate', args.dim, args.moe['num_experts'], bias=False),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(
                self.name,
                dim=args.dim,
                hidden_dim=4 * args.dim if args.hidden_dim == -1 else args.hidden_dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                mistral = False if args.hidden_dim == -1 else True
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(self.name, 'attention_norm', args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.name, 'ffn_norm', args.dim, eps=args.norm_eps)

    def get_submodules(self) -> Tuple[nn.Module]:
        return (self.attention, self.feed_forward, self.attention_norm, self.ffn_norm)

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: Optional[jnp.ndarray],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__(name='Transformer')
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        moe = False if params.moe is None else True
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, moe=moe))

        self.norm = RMSNorm(None, 'norm', params.dim, eps=params.norm_eps)
        self.output = nn.Linear(None, 'output', params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
        )

    def get_submodules(self) -> Tuple[nn.Module]:
        return (self.tok_embeddings, self.layers, self.norm, self.output)

    def forward(self, tokens: jnp.ndarray, start_pos: int) -> jnp.ndarray:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = jnp.full((seqlen, seqlen), float("-inf"))
            mask = jnp.triu(mask, k=1)
            mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.bfloat16)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output