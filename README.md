🧠 Med42-8B AWQ Quantization Evaluation Summary
📌 Model
Base Model: m42-health/Llama3-Med42-8B
Quantized Model: hoho0106tw/Femh_Pruning_med428B_awq-model
Quantization Method: AWQ (Activation-aware Weight Quantization)
📊 1. Quantization Quality (Loss-based Evaluation)

根據 loss 評估腳本 ：

🔹 Overall Metrics
Model	Loss	Perplexity
FP16	3.7863	44.09
AWQ	3.8388	46.47
🔹 Degradation
Δ Loss = +0.0525
✅ Interpretation
Δ < 0.1 → Excellent quantization
AWQ 僅造成極小數值偏移
模型語言建模能力幾乎完整保留
📉 Loss Distribution Analysis

從圖表觀察：

FP16 與 AWQ 分布高度重疊
無明顯 tail explosion（沒有嚴重錯誤樣本增加）
AWQ 僅輕微右移（合理現象）

👉 結論：

Quantization preserves token-level predictive distribution
🧪 2. Task-level Evaluation (Classification)

基於分類評估腳本 ：

🔹 Neurology Task
Model	Accuracy
FP16	0.827
AWQ	0.820
Δ = -0.007 (~0.7%)
✅ Interpretation
幾乎無性能損失
高語意任務仍維持穩定
AWQ 對 fine-grained reasoning 影響極小
🌍 3. Generalization Evaluation (跨資料集)

基於泛化測試腳本 ：

🔹 Cross-domain Task (150 samples)
Model	Accuracy
FP16	0.833
AWQ	0.807
Δ = -0.027 (~2.7%)
📊 Error Perspective
FP16: ~125 correct
AWQ: ~121 correct
差異：約 4 題 / 150 samples
✅ Interpretation

👉 在 未見資料（OOD）下

僅 2~3% degradation
無 catastrophic failure
保持 >0.8 accuracy
Strong generalization retained after quantization
💾 4. Memory Efficiency

根據記憶體測試腳本 ：

Model	GPU Memory
FP16	15.18 GB
AWQ	5.63 GB
🔹 Compression Ratio
Reduction: 2.70x
✅ Interpretation
記憶體下降約 63%
大幅降低部署門檻
支援單卡推論 / 邊緣部署
Significant memory reduction with minimal accuracy loss
🎯 5. Key Insights
🔥 Insight 1: Quantization Robustness
Δ < 3% across tasks → Highly robust quantization
🔥 Insight 2: Task Dependency
Task Type	Δ
Neurology (fine-grained)	-0.007
General (cross-domain)	-0.027

👉 說明：

AWQ 對 pattern-based 任務更穩
對 複雜推理略敏感（但仍可接受）
🔥 Insight 3: Label Engineering Impact

透過 label 清理（移除共病類）：

Accuracy: 0.45 → 0.83

👉 結論：

Problem formulation > Model selection
🚀 6. Final Conclusion
The AWQ-quantized Med42-8B model demonstrates:

✔ Excellent quantization quality (Δ loss < 0.1)
✔ Minimal performance degradation (≤ 3%)
✔ Strong generalization capability
✔ Significant memory reduction (~2.7x)

Overall, AWQ provides a highly efficient compression strategy
while preserving both language modeling and downstream task performance.
