# Forward or Reverse KL? Exploring On-Policy Distillation for Speculative Decoding

This repository is based on [DistillSpec](https://arxiv.org/abs/2310.08461): Improving Speculative Decoding via Knowledge Distillation (Zhou et. al.).

---
**Course:** Columbia University COMS4705 (Natural Language Processing) 

## 📖 Overview

This project investigates the impact of different divergence objectives-Forward KL (FKL), Reverse KL (RKL), and Jensen-Shannon Divergence (JSD)-on training draft models for **Speculative Decoding (SD)**.

Standard off-policy distillation suffers from exposure bias. We utilize **On-Policy Distillation**, where the student learns from its own generated trajectories scored by a frozen teacher . We demonstrate that the optimal divergence metric is highly dependent on the **entropy profile** of the downstream task.

## Batched Speculative Decoding for Fast Eval

A speculative decoding framework which supports batched inputs was implemented first to enable fast evaluation. Implementation is available at [batched_specdec](https://github.com/r-rishabh-j/batched_specdec) and is added as a submodule.

## 🚀 Key Findings

Our experiments show that **Mode-Seeking** behavior is preferred for reasoning, while **Mean-Seeking/Balanced** behavior is preferred for open-ended generation .

| Task Type | Dataset | Teacher Entropy | Best Metric | Insight |
| --- | --- | --- | --- | --- |
| **Math Reasoning** | GSM8k | Low (Deterministic) | **Reverse KL (RKL)** | RKL forces the student to "snap" to the correct reasoning path, ignoring valid but unaligned tokens.|
| **Summarization** | CNN/DM | High (Ambiguous) | **JSD** | JSD balances coverage and precision, preventing collapse in high-entropy regions.|

### Quantitative Results (Token Acceptance Rate )

* **Qwen3 (GSM8k):** RKL achieved **53.93%**, outperforming Baseline (49.52%) and FKL.

* **SmolLM (CNN/DM):** JSD achieved **55.35%**, outperforming RKL and FKL.

## 🛠️ Methodology

Models 

* **Math:** Teacher: `Qwen3-4B-Instruct` | Student: `Qwen3-0.6B-Instruct`
* **Summarization:** Teacher: `SmolLM-1.7B-Instruct` | Student: `SmolLM-360M-Instruct`

### Distillation Approach

We use a white-box, token-level distillation framework using **Hugging Face TRL's `GKDTrainer**`.

* **Zero-Label Training:** No ground-truth dataset labels were used; training relied entirely on student-generated trajectories scored by the teacher.


* **Metrics:** We evaluate using Token-Level and Sequence-Level Acceptance Rates.


## 💻 Implementation Details

The project includes a custom evaluation harness for **Speculative Decoding with Dynamic Batching**.

### Requirements

* Python 3.8+
* PyTorch (with CUDA support)
* Hugging Face `transformers`, `trl`, `bitsandbytes`

<!--
### Training
Training is performed on a single H100 GPU using 8-bit quantization for Qwen models .

```bash
# Example command (conceptual based on report)
python train.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --teacher_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset "gsm8k" \
    --divergence "rkl" \ # Options: fkl, rkl, jsd
    --on_policy True

```

### Evaluation

We utilize a custom SD implementation capable of handling jagged sequence lengths (dynamic batching) to measure acceptance rates.

```bash
python eval_speculative.py \
    --draft_model_path ./checkpoints/rkl_model \
    --target_model_path Qwen/Qwen2.5-7B-Instruct \
    --gamma 5 

```
-->
## 📊 Analysis

We analyze performance relative to **Teacher Entropy**.

* **Low Entropy ():** RKL dominates, maximizing precision.


* **High Entropy ():** RKL degrades; JSD remains robust where diversity is required .
