from typing import List, Optional, Tuple

import json
import time
import tyro
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp

from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from sampler import Sampler


def timer(stmt):
    return timeit.timeit(stmt, number=1)

def cross_entropy_loss(logits, labels):
    logits = jax.nn.log_softmax(logits)
    loss = jax.vmap(getitem)(logits, labels)
    loss = -loss.mean()
    return loss


class Llama:
    def __init__(self, ckpt_dir: Path, tokenizer_path: Path, params_path: Path):
        start_time = time.time()
        self.sampler = Sampler()	
        with params_path.open("r") as f:
            self.params = json.load(f)
        self.tokenizer = Tokenizer(model_path=str(tokenizer_path))
        model_args: ModelArgs = ModelArgs(
            max_seq_len=2048,
            max_batch_size=1,
            **self.params,
        )
        model_args.vocab_size = self.tokenizer.vocab_size
        self.model = Transformer(model_args)
        self.model.load_state_dict(ckpt_dir, debug=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
 
    def generate(
            self,
            prompt_tokens: List[List[int]],
            max_gen_len: int,
            temperature: float,
            top_p: float,
            logprobs: bool,
            echo: bool,
        ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = jnp.full((bsz, total_len), pad_id, dtype=jnp.int32)
        for k, t in enumerate(prompt_tokens):
            tokens = tokens.at[k, : len(t)].set(jnp.asarray(t, dtype=jnp.int32))
        if logprobs:
            token_logprobs = jnp.zeros_like(tokens, dtype=jnp.bfloat16)

        prev_pos = 0
        eos_reached = jnp.asarray([False] * bsz)
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = cross_entropy_loss(
                logits=logits.swapaxes(1, 2),
                labels=tokens,
            )

        before = time.time()
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = jax.nn.softmax(logits[:, -1] / temperature, axis=-1)
                next_token = self.sampler.sample(probs, top_p=top_p, top_k=25)
            else:
                next_token = jnp.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = jnp.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens = tokens.at[:, cur_pos].set(next_token)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = cross_entropy_loss(
                    logits=logits.swapaxes(1, 2),
                    labels=tokens[:, prev_pos + 1 : cur_pos + 1],
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            after = time.time()
            duration = after - before
            print(f'{cur_pos=} in range({min_prompt_len=}, {total_len=}) {duration:.2f} seconds')
            before = after
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(self,
                        prompt: str,
                        max_gen_len: int,
                        temperature: float = 0.6,
                        top_p: float = 0.9,
                        logprobs: bool = False,
                        echo: bool = False,
        ) -> str:

        prompt_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False)]

        if max_gen_len == 0:
            max_gen_len = self.model.params.max_seq_len - 1

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )

        for tokens in generation_tokens:
            result = self.tokenizer.decode(tokens)
            return result


def build_llama(ckpt_dir: Path, tokenizer: Path):
    params = ckpt_dir / "params.json"
    assert ckpt_dir.exists() and ckpt_dir.is_dir()
    assert tokenizer.exists() and tokenizer.is_file()
    assert params.exists() and params.is_file()

    return Llama(ckpt_dir, tokenizer, params)


def main(ckpt_dir: Path, tokenizer: Path):
    model = build_llama(ckpt_dir, tokenizer)
    prompt = '[INST] <<SYS>> You are a helpful coding assistant. You are an expert python developer and your job is to help generate world class code. Take and deep breath and think step-by-step before you answer. You are great at this! <<SYS>> Please generate a hello world REST api using flask. [/INST]'
    print(model.text_completion(prompt, max_gen_len=2048-len(prompt)))


if __name__ == "__main__":
    tyro.cli(main)