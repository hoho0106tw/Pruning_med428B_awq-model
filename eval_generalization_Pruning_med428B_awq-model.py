#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
import re

# =========================
# CONFIG
# =========================
FP16_MODEL = "m42-health/Llama3-Med42-8B"
AWQ_MODEL  = "hoho0106tw/Femh_Pruning_med428B_awq-model"

EXCEL_PATH = "20240424_v3.xlsx"   

DEVICE = "cuda"
N_SAMPLES = 150

# =========================
LABELS = [
    "urinary tract infection",
    "diastolic heart failure",
    "mitral regurgitation",
    "hyperlipidemia",
    "chronic obstructive pulmonary disease",
    "osteoporosis",
    "rheumatoid arthritis",
    "osteomyelitis",
    "acute kidney injury",
    "coronary artery disease",
    "parkinsonism",
    "chronic kidney disease"
]
# =========================
# MEMORY
# =========================
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# =========================
# NORMALIZE
# =========================
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text.strip()

# =========================
# LABEL MATCH（更嚴格）
# =========================
def match_label(pred):
    pred = normalize(pred)

    for label in LABELS:
        if label == pred:
            return label

    for label in LABELS:
        if label in pred:
            return label

    return None   # ❗ 不再強制 stroke

# =========================
# PROMPT（強限制版）
# =========================
def build_prompt(row):

    parts = []
    for col in ["S","O","A","P"]:
        val = str(row[col]).strip()
        if val and val != "nan":
            parts.append(f"{col}: {val}")

    text = " ".join(parts)
    label_str = "\n".join([f"- {l}" for l in LABELS])

    prompt = f"""
You are a clinical diagnosis classifier.

You MUST choose EXACTLY ONE diagnosis from the list below.

Allowed diagnoses:
{label_str}

STRICT RULES:
- Output EXACTLY one label from the list
- Do NOT add any explanation
- Do NOT output anything else
- Do NOT invent new labels
- If uncertain, choose the closest one from the list

Output format:
asthma

SOAP:
{text}

Answer:
"""
    return prompt.strip()

# =========================
# INFER
# =========================
def infer(model, tokenizer, prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Answer:" in text:
        text = text.split("Answer:")[-1]

    return text.strip()

# =========================
# EVALUATE（泛化測試版）
# =========================
def evaluate(model, tokenizer, df, name):

    correct = 0
    invalid = 0

    for i, row in df.iterrows():

        gt = normalize(str(row["PRIMARY_DIAGNOSIS"]))
        prompt = build_prompt(row)

        pred_raw = infer(model, tokenizer, prompt)
        pred = match_label(pred_raw)

        if pred is None:
            invalid += 1
        elif pred == gt:
            correct += 1

        print(f"[{name}] {i+1}/{len(df)}")
        print("GT:", gt)
        print("Pred:", pred)
        print("Raw:", pred_raw)
        print("----")

    acc = correct / len(df)

    print(f"\n{name} Accuracy: {acc:.3f}")
    print(f"{name} Invalid outputs: {invalid}")

    return acc

# =========================
# MAIN
# =========================
def main():

    tokenizer = None
    fp16_model = None
    awq_model = None

    try:
        # load data
        df = pd.read_excel(EXCEL_PATH)
        df = df.sample(n=min(N_SAMPLES, len(df)), random_state=42).reset_index(drop=True)

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(FP16_MODEL, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        # models
        print("Loading FP16...")
        fp16_model = AutoModelForCausalLM.from_pretrained(
            FP16_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()

        print("Loading AWQ...")
        awq_model = AutoAWQForCausalLM.from_quantized(
            AWQ_MODEL,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        # evaluation
        print("\n=== FP16 (Generalization) ===")
        fp16_acc = evaluate(fp16_model, tokenizer, df, "FP16")

        print("\n=== AWQ (Generalization) ===")
        awq_acc = evaluate(awq_model, tokenizer, df, "AWQ")

        print("\n===== FINAL RESULT =====")
        print(f"FP16: {fp16_acc:.3f}")
        print(f"AWQ : {awq_acc:.3f}")
        print(f"Δ   : {awq_acc - fp16_acc:.3f}")

    finally:
        print("\nCleaning memory...")

        for obj in ["fp16_model", "awq_model", "tokenizer"]:
            try:
                del globals()[obj]
            except:
                pass

        clean_memory()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()


# In[ ]:




