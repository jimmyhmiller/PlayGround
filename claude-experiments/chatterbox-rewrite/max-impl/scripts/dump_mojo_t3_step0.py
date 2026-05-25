"""Dump Mojo T3 step-0 cond+uncond logits and compare to upstream dump.

We run our full op_t3 pipeline but stop after the first sample. We snoop on
the cfg-combine call to capture cond, uncond, and cfg-combined logits.

Approach: rather than instrument Mojo, we hijack the orchestrator's T3
inputs and just read the raw logits buffer from Mojo before sampling.

We use the existing op_t3 infrastructure: prepare conditionals via the
Mojo wrapper, then call op_t3 with text="Hello world." But to extract
step 0 logits we need to add a debug hook.

Simpler approach: write a Mojo test that produces the same prefix, runs
prefill, and saves the (B2, V) raw logits buffer.
"""
import os, sys
import numpy as np
import torch

# Just verify upstream dump exists.
up = np.load("/tmp/t3_dump/upstream_logits.npz")
print(f"upstream text_ids: {up['text_ids'].tolist()}")
print(f"upstream sampled[:10]: {up['sampled_tokens'][:10].tolist()}")
print(f"upstream cond[0, top5]: {np.argsort(up['logits_cond'][0])[-5:][::-1].tolist()}")
print(f"upstream cfg[0, top5]:  {np.argsort(up['logits_cfg'][0])[-5:][::-1].tolist()}")
print(f"upstream step 0 cond logits: max={up['logits_cond'][0].max():.3f}  min={up['logits_cond'][0].min():.3f}")
