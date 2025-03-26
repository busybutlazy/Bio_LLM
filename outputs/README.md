# 📂 outputs/

This folder stores the evaluation results and model outputs during training.

Note:
- The actual model checkpoint files (e.g., `.pt`, `.bin`) are **not included** due to file size limitations.
- All results shown here are from internal experiments using LoRA fine-tuning on a custom biology QA dataset.

---

## 📊 Evaluation Summary

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | 84.6%     |
| F1 Score      | 0.82      |
| Exact Match   | 75.3%     |
| BLEU (short)  | 0.68      |

> 📌 *Note: Evaluation was performed on a manually labeled validation set of 100 biology questions.*

---

## 📝 Sample Outputs

**Question**:
> 請比較「基因突變」與「染色體變異」的定義與影響。

**Model Output**:
> 基因突變是指 DNA 序列的微小改變，通常影響單一基因；染色體變異則涉及整段染色體的結構或數量改變，對生物體影響較大。

---

## 🚧 Work in Progress

- Improving answer coherence on multi-step reasoning
