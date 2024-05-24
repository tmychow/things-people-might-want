from pathlib import Path

import torch
import jax
import jax.numpy as jnp

import tyro
import json
import sentencepiece
import sentencepiece.sentencepiece_model_pb2 as model

from transformers import AutoModelForCausalLM, AutoTokenizer

def translate_key(in_key: str):
    out_key = in_key.replace('.weight', '')
    if out_key.startswith('model.'):
        out_key = out_key.replace('model.', '')
        if out_key.endswith('input_layernorm'):
            out_key = out_key.replace('input_layernorm', 'attention_norm')
        elif out_key.endswith('mlp.down_proj'):
            out_key = out_key.replace('mlp.down_proj', 'feed_forward.w2')
        elif out_key.endswith('mlp.gate_proj'):
            out_key = out_key.replace('mlp.gate_proj', 'feed_forward.w1')
        elif out_key.endswith('mlp.up_proj'):
            out_key = out_key.replace('mlp.up_proj', 'feed_forward.w3')
        elif out_key.endswith('post_attention_layernorm'):
            out_key = out_key.replace('post_attention_layernorm', 'ffn_norm')
        elif out_key.endswith('self_attn.k_proj'):
            out_key = out_key.replace('self_attn.k_proj', 'attention.wk')
        elif out_key.endswith('self_attn.o_proj'):
            out_key = out_key.replace('self_attn.o_proj', 'attention.wo')
        elif out_key.endswith('self_attn.q_proj'):
            out_key = out_key.replace('self_attn.q_proj', 'attention.wq')
        elif out_key.endswith('self_attn.v_proj'):
            out_key = out_key.replace('self_attn.v_proj', 'attention.wv')
        elif out_key == 'embed_tokens':
            out_key = 'tok_embeddings'
        elif out_key == 'norm':
            out_key = 'norm'
        else:
            print(f"Don't know how to handle {in_key=}")
    elif out_key == 'lm_head':
        out_key = 'output'
    else:
        print(f"Don't know how to handle {in_key=}")
    return f'{out_key}.weight'


def reverse_permute(tensor: torch.Tensor, n_heads: int = 32, dim1:int = 4096, dim2: int = 4096) -> torch.Tensor:
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def main(model_id: str, out_dir: Path):
    #t_path = Path.home() / '.hf_token'
    #token = t_path.read_text().strip()
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    #tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()
    tokenizer.save_pretrained(out_dir)
    m = model.ModelProto()
    m.ParseFromString(open(f'{out_dir}/tokenizer.model', "rb").read())
    with open(f'{out_dir}/tokenizer_config.json', "r") as f:
        config = json.load(f)
    p = list(m.pieces)
    m.ClearField('pieces')
    size = len(p)
    for token_id, token_data in config["added_tokens_decoder"].items():
        new_token = model.ModelProto().SentencePiece()
        new_token.piece = token_data['content']
        new_token.score = 0
        if 'unk' in token_data['content']:
            new_token.type = model.ModelProto.SentencePiece.Type.UNKNOWN
        else:
            new_token.type = model.ModelProto.SentencePiece.Type.CONTROL
        idx = int(token_id)
        if idx > size-1: 
            p.append(new_token)
        else:
            old_token = p[idx]
            p[idx] = new_token
    # Deal with 'additional_special_tokens'
    m.pieces.extend(p)
    with open(f'{out_dir}/tokenizer.model', 'wb') as f:
        f.write(m.SerializeToString())

    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
    #hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, offload_folder="weights/tmp/offload", token=token)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        for hf_name, param in state_dict.items():
            name = translate_key(hf_name)
            if name.endswith('wq.weight'):
                param = reverse_permute(param, n_heads=32, dim1=4096, dim2=4096) # 7B
                #param = reverse_permute(param, n_heads=64, dim1=8192, dim2=8192)   # 70B
            elif name.endswith('wk.weight'): #wk.weight
                param = reverse_permute(param, n_heads=8, dim1=1024, dim2=4096)  # 7B
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=8192)    # 70B
            else:
                pass
            f32_out = param.cpu().numpy()
            bf16_out = jnp.asarray(f32_out, dtype=jnp.bfloat16).reshape(*param.shape)
            print(f'Writing {hf_name} as {name} to {out_dir}/{name}.npy')
            jnp.save(f'{out_dir}/{name}.npy', bf16_out)  # Detach, convert to CPU, and save as NumPy array


if __name__ == "__main__":
    tyro.cli(main)