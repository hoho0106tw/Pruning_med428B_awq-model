#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 🔥 關鍵

import torch
from transformers import AutoModelForCausalLM
from awq import AutoAWQForCausalLM

FP16_MODEL = "m42-health/Llama3-Med42-8B"
AWQ_MODEL  = "hoho0106tw/Femh_Pruning_med428B_awq-model"

DEVICE = torch.device("cuda:0")

def measure_memory(load_fn, name):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\nLoading {name}...")

    model = load_fn().to(DEVICE)

    print("Model device:", next(model.parameters()).device)

    dummy = torch.ones((1, 512), dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        _ = model(input_ids=dummy)

    mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"{name} GPU Memory: {mem:.2f} GB")

    del model
    torch.cuda.empty_cache()

    return mem


# FP16
fp16_mem = measure_memory(
    lambda: AutoModelForCausalLM.from_pretrained(
        FP16_MODEL,
        torch_dtype=torch.float16
    ).eval(),
    "FP16"
)

# AWQ
awq_mem = measure_memory(
    lambda: AutoAWQForCausalLM.from_quantized(
        AWQ_MODEL,
        trust_remote_code=True
    ).eval(),
    "AWQ"
)

print("\n===== SUMMARY =====")
print(f"FP16: {fp16_mem:.2f} GB")
print(f"AWQ : {awq_mem:.2f} GB")
print(f"Reduction: {fp16_mem / awq_mem:.2f}x")


# In[ ]:




