# TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking

## Introduction

Welcome to **TFRank** 🎉
We propose a training framework **TFRank** for small-scale LLMs that enables **efficient and effective pointwise reasoning ranking** *without explicit reasoning output at inference*.

<div align="center">
  <img src="figure/framework.png">
</div>

---

## 📦 Installation

Clone repository:

```bash
git clone https://github.com/JOHNNY-fans/TFRank.git
cd TFRank
pip install -r requirements.txt
```

---

## Resources

### 📦 Models
| Model | Description |
|:---------|:------------|
| [TFRank-SFT-Qwen2.5-7B-Instruct](https://huggingface.co/Johnnyfans/TFRank-SFT-Qwen2.5-7B-Instruct) | Built on Qwen2.5-7B-Instruct with full SFT data. |
| [TFRank-SFT-Qwen3-8B](https://huggingface.co/Johnnyfans/TFRank-SFT-Qwen3-8B) | Built on Qwen3 8B with full SFT data.|
| [TFRank-SFT-GRPO-Qwen3-8B](https://huggingface.co/Johnnyfans/TFRank-SFT-GRPO-Qwen3-8B) | Built on Qwen3 8B with full SFT plus a small set of GRPO data.|
| [TFRank-GRPO-Qwen3-0.6B](https://huggingface.co/Johnnyfans/TFRank-GRPO-Qwen3-0.6B) | Built on Qwen3 0.6B and trained on full GRPO data. Efficiency oriented (high throughput, low latency). Suited for large scale online reranking and high concurrency. |
| [TFRank-GRPO-Qwen3-1.7B](https://huggingface.co/Johnnyfans/TFRank-GRPO-Qwen3-1.7B) | Built on Qwen3 1.7B and trained on full GRPO data. |
| [TFRank-GRPO-Qwen3-4B](https://huggingface.co/Johnnyfans/TFRank-GRPO-Qwen3-4B) | Built on Qwen3 4B and trained on full GRPO data. |
| [TFRank-GRPO-Qwen3-8B](https://huggingface.co/Johnnyfans/TFRank-GRPO-Qwen3-8B) | Built on Qwen3 8B and trained on full GRPO data. |

Additional, stronger models will be released progressively. *To be released soon...*

### 📂 Datasets

We provide high-quality datasets constructed from multiple sources, integrating **multi-task supervision**, **reasoning chains (CoT)**, and **think-mode-swift** training samples.

| Dataset                                                                                           | Description                                                                                                                                                                                                                                                        |
| :------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [TFRank-sft-training-data](https://huggingface.co/datasets/Johnnyfans/TFRank-sft-training-data)   | Supervised fine-tuning (SFT) samples aggregated from [Rank1](https://github.com/orionw/rank1), MS MARCO, and DeepSeek-R1. Includes multi-task, CoT, and think-mode-swift samples. |
| [TFRank-grpo-training-data](https://huggingface.co/datasets/Johnnyfans/TFRank-grpo-training-data) | GRPO-based training samples from the same sources.                                                                                                                                                            |

---

## Performance
<div align="center">
  <img src="figure/size_efficiency_performance.png" width="720px">
  <p><em>
    Size and efficiency trade-offs for ranking performance on the BRIGHT benchmark.  
    (a) NDCG@10 versus model size for different ranker families;  
    (b) NDCG@10 versus processed queries per hour (efficiency).  
    All TFRank models are trained on the Qwen3 series.
  </em></p>
</div>

---

## 🚀 Inference Quick Start

Below are two minimal examples demonstrating how to run TFRank for query–document relevance scoring.

### 1️⃣ Start a vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/tfrank_checkpoint \
    --served-model-name rele_pointwise \
    --port 8113
```

---

### 2️⃣ Initialize the Ranker

```python
from evaluation.minimal_ranker import TFRankDemoRanker

ranker = TFRankDemoRanker(
    model_name="/path/to/your/tfrank_checkpoint",
    api_base="http://localhost:8113/v1",
    api_key="any-string",          # vLLM usually ignores this
    think_mode=False,              # set True to enable /think reasoning
    reasoning_model=False,         # set True if using a reasoning-head model
)
```

---

### 📝 Example 1 — Completely Irrelevant Document

```python
query = "what nano means"

document = "What does nano mean? Nano means very, very small. When it comes to making your body work, nano-materials are very important. A nanometre is one millionth of a millimetre. Your fingernail is about one millimetre thick. There are a lot of nano-materials making up your finger nail! Nanotechnology scientists move atoms and molecules around to make amazing new technologies. Nanotechnology is already in products like sunscreen."


final_score, fg_score, yes_score, response = ranker.score(query, document)

print("Final relevance score (0–1):", final_score) # 0.9997
print("Fine-grained score (normalized):", fg_score)
print("Yes-probability:", yes_score)
print("\nModel response:\n", response) # yes(4)
```

---

### 📝 Example 2 — Highly Relevant Document

```python
query = "what is a musket?"

document = "8 Unusual Civil War Weapons You might think the Civil War was only fought with muskets, bayonets and cannons, but those weren’t the only deadly weapons to haunt the battlefields of the 1860s."


final_score, fg_score, yes_score, response = ranker.score(query, document)

print("Final relevance score (0–1):", final_score) # 0.1228
print("Fine-grained score (normalized):", fg_score)
print("Yes-probability:", yes_score)
print("\nModel response:\n", response) # no(1)
```

---

## 📓 Full Notebook Demo

A full inference notebook is available at:

```
evaluation/inference_demo.ipynb
```

---

# 📚 Citation

If you use TFRank in your research, please cite:

```bibtex
@article{fan2025tfrank,
  title={TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking},
  author={Fan, Yongqi and Chen, Xiaoyang and Ye, Dezhi and Liu, Jie and Liang, Haijin and Ma, Jin and He, Ben and Sun, Yingfei and Ruan, Tong},
  journal={arXiv preprint arXiv:2508.09539},
  year={2025}
}
```

---
