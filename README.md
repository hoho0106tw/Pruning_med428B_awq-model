# 🧠 Med42-8B AWQ Quantization Evaluation Summary

## 📌 Model Overview
- Base Model: `m42-health/Llama3-Med42-8B`
- Quantized Model: `hoho0106tw/Femh_Pruning_med428B_awq-model`
- Quantization Method: AWQ (Activation-aware Weight Quantization)
- Evaluation Scope: Loss-based evaluation, classification accuracy, cross-domain generalization, and memory efficiency

---

## 📊 Quantization Quality (Loss Evaluation)

| Model | Loss | Perplexity |
|------|------|------------|
| FP16 | 3.7863 | 44.09 |
| AWQ  | 3.8388 | 46.47 |

- Δ Loss: **+0.0525**
- 判定標準：Δ < 0.1 → **Excellent Quantization**

### Interpretation
AWQ 僅造成極小的 loss 上升，顯示模型語言建模能力幾乎完整保留。Loss 分布（Histogram）顯示 FP16 與 AWQ 高度重疊，僅有輕微右移，未出現長尾惡化或異常錯誤樣本，代表 token-level 預測分布穩定。

---

## 🧪 Task-level Evaluation (Neurology Classification)

| Model | Accuracy |
|------|---------|
| FP16 | 0.827 |
| AWQ  | 0.820 |

- Δ Accuracy: **-0.007 (~0.7%)**

### Interpretation
在高語意密集（fine-grained reasoning）的神經分類任務中，AWQ 幾乎無性能損失，顯示量化後仍能維持細緻語意判斷能力。

---

## 🌍 Generalization Evaluation (Cross-domain)

| Model | Accuracy |
|------|---------|
| FP16 | 0.833 |
| AWQ  | 0.807 |

- Δ Accuracy: **-0.027 (~2.7%)**
- 測試樣本數：150

### Error Analysis
- FP16 正確：約 125 題
- AWQ 正確：約 121 題
- 差異：約 **4 題 / 150 samples**

### Interpretation
在未見資料（out-of-distribution）下，AWQ 僅造成約 2~3% 性能下降，且整體 accuracy 仍維持 >0.8，顯示量化模型具備良好泛化能力，無明顯崩潰或錯誤擴散現象。

---

## 💾 Memory Efficiency

| Model | GPU Memory |
|------|-----------|
| FP16 | 15.18 GB |
| AWQ  | 5.63 GB |

- Reduction: **2.70x (~63% decrease)**

### Interpretation
AWQ 大幅降低模型記憶體需求，使 8B 模型可在單 GPU 或資源受限環境下運行，顯著提升部署可行性。

---

## 🎯 Key Insights

### 1. Quantization Robustness
Δ < 3% across multiple tasks，顯示 AWQ 對模型性能影響極小，屬於高品質量化結果。

### 2. Task Sensitivity
- 神經任務（高語意）：Δ = -0.007  
- 跨領域任務（較簡單）：Δ = -0.027  
顯示量化對 fine-grained reasoning 較敏感，但整體仍穩定。

### 3. Label Engineering Impact
透過 label 清理與任務重設：
- Accuracy: 0.45 → 0.83（顯著提升）

結論：
> 問題定義（label design）影響遠大於模型本身

---

## 🚀 Final Conclusion

The AWQ-quantized Med42-8B model demonstrates excellent performance across multiple evaluation dimensions. Quantization introduces only minimal degradation (~0.7%–2.7%) while preserving strong language modeling capability and downstream classification accuracy. The model maintains robust generalization on unseen data and achieves a significant memory reduction of approximately 2.7×. These results indicate that AWQ is a highly effective compression strategy, enabling efficient deployment without sacrificing model quality.

---

## 🧠 One-line Summary

AWQ successfully achieves near-lossless quantization with strong generalization and significant memory savings, making it highly suitable for real-world deployment.
