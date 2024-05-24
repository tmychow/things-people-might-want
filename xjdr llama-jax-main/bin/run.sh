#!/usr/bin/env bash

docker run -it -v weights:/weights llama-jax:latest -- /app/generation.py --ckpt-dir /weights --tokenizer /weights/tokenizer.model