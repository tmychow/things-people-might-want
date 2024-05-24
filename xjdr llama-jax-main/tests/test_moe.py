from typing import List, Optional, Tuple

from dataclasses import dataclass

from nn import Linear, Module, ModuleList
from model import FeedForward

import torch
import torch.nn as nn
import torch.nn.functional as F 

import jax
import jax.numpy as jnp

import numpy as np


@dataclass
class MoeArgs:
    num_experts: int
    num_experts_per_tok: int


class TorchFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TorchMoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            w = weights[batch_idx, nth_expert, None]
            e = expert(inputs[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
        return results


class JaxMoeLayer(Module):
    def __init__(self, prefix: str, experts: List[Module], gate: Module, moe_args: MoeArgs):
        super().__init__(name=Module.add_prefix(prefix, "feed_forward.experts"), needs_weight=True),
        assert len(experts) > 0
        self.experts = ModuleList()
        self.gate = gate
        self.args = moe_args
        for expert in experts:
            self.experts.append(expert)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        gate_logits = self.gate(x)
        weights, selected_experts = jax.lax.top_k(gate_logits, 2)
        weights = jax.nn.softmax(weights, axis=1).astype(jnp.float32)
        results = jnp.zeros_like(x)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = jnp.where(selected_experts == i)
            results = results.at[batch_idx].set(results[batch_idx] + weights[batch_idx, nth_expert, None] * expert.forward(x[batch_idx]))
        return results

@torch.no_grad
def test():
    seqlen = 12
    n = 4096
    d = 14336
    gn = 4096
    gd = 4096
    moe_args = MoeArgs(num_experts=8, num_experts_per_tok=2)
    texperts = []
    jexperts = []
    
    w1_pt = torch.randn((d,n), dtype=torch.float)
    w2_pt = torch.randn((n,d), dtype=torch.float)
    w3_pt = torch.randn((d,n), dtype=torch.float)

    #gwn = jnp.load(file='layers.0.feed_forward.gate.weight.npy', mmap_mode='r', allow_pickle=True).astype(jnp.float32)

    for i in range(moe_args.num_experts):
        x_pt = torch.randn(n, dtype=torch.float).view(1, 1, n)
        texperts.append(TorchFeedForward(n, d))
        jexperts.append(FeedForward(None, n, d, 1, 1, True, True))

    for e in texperts:
        e.w1.weight = nn.Parameter(w1_pt)
        e.w2.weight = nn.Parameter(w2_pt)
        e.w3.weight = nn.Parameter(w3_pt)

    for e in jexperts:
        e.w1.w = jnp.asarray(w1_pt.numpy())
        e.w2.w = jnp.asarray(w2_pt.numpy())
        e.w3.w = jnp.asarray(w3_pt.numpy())

    tgate = nn.Linear(gn, 8, False)
    jgate = Linear(None, "gate", gn, 8)
    gw = torch.randn((8,n), dtype=torch.float)
    tgate.weight = nn.Parameter(torch.from_numpy(np.array(gw)))
    jgate.w = jnp.asarray(gw)
    tmoe = TorchMoeLayer(texperts, tgate, moe_args)
    jmoe = JaxMoeLayer("jmoe", jexperts, jgate, moe_args)


    x_pt = torch.randn((seqlen, n), dtype=torch.float)
    x_jax = jnp.asarray(x_pt.numpy()).astype(jnp.float32)

    out_pt = tmoe.forward(x_pt)
    out_jax = jmoe.forward(x_jax)
    out_jax = jmoe.forward(x_jax)

    #print(f'{out_pt[:30]=}')
    #print(f'{out_jax[:30]=}')
    assert jnp.allclose(out_pt.numpy(), out_jax, rtol=1e-01, atol=1e-01), seqlen


if __name__ == '__main__':
    test()