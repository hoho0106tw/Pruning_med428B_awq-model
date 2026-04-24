#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
from huggingface_hub import login
from huggingface_hub.utils import HfFolder

# =========================
# HF TOKEN
# =========================
def _token_is_valid(token):
    return token is not None and isinstance(token, str) and len(token) > 0

def _load_hf_token():
    try:
        return HfFolder.get_token()
    except:
        return None

env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
file_token = _load_hf_token()
hf_token = env_token if _token_is_valid(env_token) else file_token

if hf_token:
    login(token=hf_token)

# =========================
# CONFIG
# =========================
FP16_MODEL = "m42-health/Llama3-Med42-8B"
AWQ_MODEL  = "hoho0106tw/Femh_Pruning_med428B_awq-model"
EXCEL_PATH = "sample_200_v4.xlsx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 2048

# =========================
# TOKENIZER
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FP16_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD MODELS
# =========================
print("Loading FP16 model...")
fp16_model = AutoModelForCausalLM.from_pretrained(
    FP16_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

print("Loading AWQ model...")
awq_model = AutoAWQForCausalLM.from_quantized(
    AWQ_MODEL,
    device_map="auto",
    trust_remote_code=True
).eval()

# =========================
# BUILD DATA（S / O / AP）
# =========================
def build_test_data():
    df = pd.read_excel(EXCEL_PATH)

    texts = []
    labels = []

    for _, row in df.iterrows():

        s = "" if pd.isna(row.get("S")) else str(row.get("S")).strip()
        texts.append(f"S: {s}")
        labels.append("S")

        o = "" if pd.isna(row.get("O")) else str(row.get("O")).strip()
        texts.append(f"O: {o}")
        labels.append("O")

        a = "" if pd.isna(row.get("A")) else str(row.get("A")).strip()
        p = "" if pd.isna(row.get("P")) else str(row.get("P")).strip()
        texts.append(f"A: {a} P: {p}")
        labels.append("AP")

    print(f"Total chunks: {len(texts)}")
    return texts, labels

# =========================
# LOSS
# =========================
def compute_loss(model, texts):
    losses = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean"
        )

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, ppl, losses

# =========================
# RUN
# =========================
texts, labels = build_test_data()

print("\n=== Evaluating FP16 ===")
fp16_loss, fp16_ppl, fp16_all = compute_loss(fp16_model, texts)

print("\n=== Evaluating AWQ ===")
awq_loss, awq_ppl, awq_all = compute_loss(awq_model, texts)

# =========================
# RESULT
# =========================
delta = awq_loss - fp16_loss

print("\n===== RESULT =====")
print(f"FP16 Loss: {fp16_loss:.4f}, PPL: {fp16_ppl:.2f}")
print(f"AWQ  Loss: {awq_loss:.4f}, PPL: {awq_ppl:.2f}")
print(f"Delta: {delta:.4f}")

if delta < 0.1:
    print("✅ Excellent quantization")
elif delta < 0.3:
    print("👍 Good quantization")
else:
    print("⚠️ Degradation detected")

# =========================
# ANALYSIS
# =========================
df = pd.DataFrame({
    "type": labels,
    "fp16_loss": fp16_all,
    "awq_loss": awq_all,
    "delta": [a - b for a, b in zip(awq_all, fp16_all)]
})

print("\n=== Avg delta by type ===")
print(df.groupby("type")["delta"].mean())

# =========================
# LOSS 分布（Histogram）
# =========================
plt.figure(figsize=(8,5))

plt.hist(fp16_all, bins=30, alpha=0.5, label="FP16")
plt.hist(awq_all, bins=30, alpha=0.5, label="AWQ")

plt.title("Loss Distribution")
plt.xlabel("Loss")
plt.ylabel("Count")
plt.legend()
plt.grid(True)

plt.show()

# =========================
# OPTIONAL：找 hardest samples
# =========================
print("\n=== Top Hard Samples ===")
df_sorted = df.sort_values("fp16_loss", ascending=False)
print(df_sorted.head(10))


# In[ ]:




