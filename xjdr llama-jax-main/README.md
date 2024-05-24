# llama-jax
Meta Llama torch code translated to jax (attempted to be line by line equivalent). Supports mistral and mixtral models as well. Download the weights from huggingface for simple use.
Naive imlementation of torch modules for consistency
This is used as a teaching tool and research repo, not intended for production use.
Currently only supports BS=1
Does not support Llama3 tokenizer yet, but it would be trivial to add if there is demand.

TODO:
 - tpu deployment via docker on GCP
 - gpu deployment on AWS
 - local cpu deployment
 - local gpu deployment
 - clean up a lot of the outdated garbage (while keeping it consistent with the OG llama code)

## Requirements
 this works on debian 11, debian 12 and ubuntu 20.04. It probably works on other things too, but its untested
 install poetry if you don't have it
 ```bash
 curl -sSL https://install.python-poetry.org | python3 -
 ```

## Running locally
```bash
poetry install --no-root
poetry run python convert_hf_weights.py --model-id teknium/OpenHermes-2.5-Mistral-7B --out-dir weights
#if you are OOMing during the conversion:
sudo echo -1000 > /proc/<convert_pid>/oom_score_adj
PYTHONPATH=. poetry run python generation.py --ckpt-dir weights --tokenizer weights/tokenizer.model
```

you need to create a weights/params.json

for example for mistral 7B:
```json
{
    "dim": 4096,
    "n_layers": 32,
    "head_dim": 128,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "sliding_window": 4096,
    "vocab_size": 32000
}
```

this is for legacy reasons for an internal codebase, but its hacky and i hate it. i will fix it soon. 


the model is currently set up for 2048 tokens of generation, but you can change by changing the max_gen_len to your desired value in generation.py:
```python
def main(ckpt_dir: Path, tokenizer: Path):
    model = build_llama(ckpt_dir, tokenizer)
    prompt = '[INST] <<SYS>> You are a helpful coding assistant. You are an expert python developer and your job is to help generate world class code. Take and deep breath and think step-by-step before you answer. You are great at this! <<SYS>> Please generate a hello world REST api using flask. [/INST]'
    print(model.text_completion(prompt, max_gen_len=<HERE>))
```

for CPU testing, i usually set the max_gen_len to 35. it usually generates at 60 sec / token (its very fast on TPU / GPU dont worry) and the output should look something like this:
```bash
(llama-jax-py3.11) ➜  llama-jax git:(main) ✗ PYTHONPATH=. poetry run python generation.py --ckpt-dir weights --tokenizer weights/tokenizer.model
Loaded in 0.75 seconds
cur_pos=72 in range(min_prompt_len=72, total_len=107) 80.89 seconds
cur_pos=73 in range(min_prompt_len=72, total_len=107) 59.27 seconds
cur_pos=74 in range(min_prompt_len=72, total_len=107) 57.97 seconds
cur_pos=75 in range(min_prompt_len=72, total_len=107) 58.38 seconds
cur_pos=76 in range(min_prompt_len=72, total_len=107) 58.09 seconds
cur_pos=77 in range(min_prompt_len=72, total_len=107) 57.96 seconds
cur_pos=78 in range(min_prompt_len=72, total_len=107) 58.33 seconds
cur_pos=79 in range(min_prompt_len=72, total_len=107) 58.35 seconds
cur_pos=80 in range(min_prompt_len=72, total_len=107) 58.06 seconds
cur_pos=81 in range(min_prompt_len=72, total_len=107) 58.00 seconds
cur_pos=82 in range(min_prompt_len=72, total_len=107) 57.77 seconds
cur_pos=83 in range(min_prompt_len=72, total_len=107) 57.98 seconds
cur_pos=84 in range(min_prompt_len=72, total_len=107) 58.01 seconds
cur_pos=85 in range(min_prompt_len=72, total_len=107) 58.00 seconds
cur_pos=86 in range(min_prompt_len=72, total_len=107) 59.10 seconds
cur_pos=87 in range(min_prompt_len=72, total_len=107) 60.54 seconds
cur_pos=88 in range(min_prompt_len=72, total_len=107) 58.47 seconds
cur_pos=89 in range(min_prompt_len=72, total_len=107) 58.44 seconds
cur_pos=90 in range(min_prompt_len=72, total_len=107) 59.31 seconds
cur_pos=91 in range(min_prompt_len=72, total_len=107) 58.53 seconds
cur_pos=92 in range(min_prompt_len=72, total_len=107) 58.66 seconds
cur_pos=93 in range(min_prompt_len=72, total_len=107) 58.43 seconds
cur_pos=94 in range(min_prompt_len=72, total_len=107) 58.09 seconds
cur_pos=95 in range(min_prompt_len=72, total_len=107) 58.42 seconds
cur_pos=96 in range(min_prompt_len=72, total_len=107) 58.78 seconds
cur_pos=97 in range(min_prompt_len=72, total_len=107) 58.42 seconds
cur_pos=98 in range(min_prompt_len=72, total_len=107) 59.09 seconds
cur_pos=99 in range(min_prompt_len=72, total_len=107) 58.58 seconds
cur_pos=100 in range(min_prompt_len=72, total_len=107) 58.57 seconds
cur_pos=101 in range(min_prompt_len=72, total_len=107) 58.77 seconds
cur_pos=102 in range(min_prompt_len=72, total_len=107) 58.64 seconds
cur_pos=103 in range(min_prompt_len=72, total_len=107) 58.72 seconds
cur_pos=104 in range(min_prompt_len=72, total_len=107) 58.06 seconds
cur_pos=105 in range(min_prompt_len=72, total_len=107) 58.62 seconds
cur_pos=106 in range(min_prompt_len=72, total_len=107) 57.54 seconds


Sure, I can help you generate a simple "Hello World" REST API using Flask, a popular web framework for building web applications in Python. 
```

the sampler is simplistic as it is a port of the meta-llama code, it would be easy to update if there is demand. 

## Running with Docker
TODO