# AttnRes_PEFT

Implementing Attention Residuals as a PEFT (parameter-efficient fine-tuning) method, with comparison to LoRA.

## Overview

This project explores whether **Attention Residuals (AttnRes)** can be adapted into a **PEFT-style method** for downstream fine-tuning.

The starting point is the paper:

- **Attention Residuals**
- arXiv: [2603.15031](https://arxiv.org/abs/2603.15031)

The original paper introduces AttnRes as a replacement for standard residual accumulation in PreNorm Transformer architectures. In this project, the goal is to preserve that core idea as much as possible while turning it into a practical PEFT method that can be trained and evaluated in a workflow similar to LoRA.

## Project Goal

The main goal is to build a clean and comparable implementation of **AttnRes-as-PEFT**, then evaluate it against **LoRA** under matched conditions.

We want to answer questions such as:

- Can AttnRes be converted into a parameter-efficient adaptation mechanism without losing its original motivation?
- How does it compare with LoRA in trainable parameter count, memory usage, runtime, and downstream performance?
- What trade-offs appear when AttnRes is used for fine-tuning rather than full pretraining?
- Can depth-wise residual routing provide a useful new adaptation axis for PEFT, even if it does not directly modify backbone weights?

## Team Contribution

| Area | Content | Owner |
|---|---|---|
| AttnRes method | Core AttnRes implementation for Qwen, hook-based integration, and AttnRes model construction. | Weiguo |
| Attention analysis | Layer analysis code and analysis utilities under `src/analyze/`. | Weiguo |
| Training and evaluation | Training pipeline, evaluation pipeline, and comparison workflow for base, LoRA, and AttnRes. | Shaowei |
| Experiment workflow | Data/model download scripts, experiment commands under `commands/`, and result collection into CSV tables. | Shaowei |

## Core IDEA

AttnRes-PEFT turns each Transformer layer into a small decision point: instead of only trusting the current layer output, the model can also "look back" across earlier hidden states and borrow the most useful representation depth-wise.

At every layer, the adapter does three things:

1. Collect previous hidden states from earlier depths.
2. Compute depth attention weights over those states with a learned query vector.
3. Build a weighted residual correction, normalize it, and add it to the frozen backbone output.

In compact form:

$$
\tilde{h}_t^{(k)}=\mathrm{RMSNorm}_{\text{score}}^{(\ell)}\!\left(h_t^{(k)}\right), \qquad
e_{k,t}^{(\ell)}=\langle q^{(\ell)}, \tilde{h}_t^{(k)} \rangle,\qquad
\alpha_{k,t}^{(\ell)}=\mathrm{softmax}_k\!\left(e_{k,t}^{(\ell)}\right)
$$

$$
\Delta h_t^{(\ell)}=\mathrm{RMSNorm}_{\text{out}}^{(\ell)}\!\left(\sum_{k < \ell} \alpha_{k,t}^{(\ell)} h_t^{(k)}\right), \qquad
h_{\text{out}}^{(\ell)} = h_{\text{base}}^{(\ell)} + g^{(\ell)} \cdot \Delta h_t^{(\ell)}
$$

where $\alpha$ is a softmax over depth and $g^{(\ell)}$ is a learnable scalar gate for layer $\ell$.

### Why this is PEFT-friendly

- The backbone model is frozen.
- Only lightweight adapter parameters are trainable (query vector, two RMSNorm scales, and gate per layer).
- Per-layer trainable parameters are exactly $3d + 1$ in the current implementation.
- This keeps trainable parameter count low while still adding expressive adaptation capacity.
- It is not necessarily memory-saving versus LoRA in runtime training settings (especially with larger lookback).

### Safe and controllable behavior

The gate is the key control knob:

- If $g=0$, the adapter contribution is exactly zero, so the model behaves like the original backbone.
- Starting from `gate_init=0.0` gives a "do-no-harm" initialization: training begins from base-model behavior and learns to add residual depth information only when useful.
- Increasing learned gate values allows stronger AttnRes influence.

The wrapper also supports practical control modes for analysis:

- `lookback` limits depth attention to recent layers for better efficiency.
- You can route generation/forward through strict base behavior when adapters are effectively disabled, making parity checks and ablations straightforward.

In short, this adapter keeps the spirit of Attention Residuals, but packages it into a clean, switchable, and parameter-efficient fine-tuning mechanism you can compare fairly against LoRA.

## Repository Structure

The codebase is organized into four main parts:

- [`report.pdf`](report.pdf): project report with full method, experiments, and analysis write-up.
- `src/AttnResAdapter.py`: core AttnRes PEFT wrapper on top of the frozen backbone.
- `src/trainer.py`: training pipeline for LoRA and AttnRes.
- `src/evaluator.py`: evaluation pipeline for base, LoRA, and AttnRes.
- `src/analyze/`: layer analysis code added for attention-based inspection.
- `src/analyze/analysis_notebook.ipynb`: notebook for RTE full-lookback analysis and cross-task attention-pattern comparison.
- `scripts/`: thin Python entrypoints.
- `commands/train/`: ready-to-run training commands.
- `commands/evaluate/`: ready-to-run evaluation commands.
- `scripts/collect_eval_results.py`: merge all evaluation summaries into CSV tables.

## Tasks

The current experiments focus on three tasks:

- `gsm8k`
- `rte`
- `boolq`

For the main comparison, each task is evaluated with:

- `base`
- `lora`
- `attnres`

In addition, `rte` is used for `lookback` ablation of AttnRes.

## Main Findings

At the current stage, the main empirical picture is:

- `LoRA` is the strongest method across the three main tasks.
- `AttnRes-PEFT` gives the clearest improvement on `gsm8k`, suggesting that depth-wise residual routing is more helpful for reasoning-heavy tasks than for short classification tasks.
- On `boolq` and `rte`, AttnRes is closer to the frozen baseline and remains below LoRA.
- In the `rte` ablation, increasing `lookback` does not change the number of trainable parameters, but it does increase runtime and memory usage.
- The `lookback` ablation suggests that performance saturates fairly early, while the attention analysis shows that different tasks reuse earlier layers in different ways.

## Main Commands

Download data and model:

```bash
bash commands/down_data_model.sh
```

Run training:

```bash
bash commands/train/train_lora_gsm8k.sh
bash commands/train/train_attenres_gsm8k.sh
```

Run evaluation:

```bash
bash commands/evaluate/evaluate_all_main.sh
bash commands/evaluate/evaluate_attnres_rte_ablation.sh
```

Merge evaluation summaries:

```bash
bash commands/merge_result.sh
```

This will generate:

- `results/main_results.csv`
- `results/ablation_results.csv`

Run analysis:

- Use `src/analyze/analysis_notebook.ipynb` for the final layer-usage analysis.
- The notebook is organized into two parts:
  - `RTE` full-lookback analysis for detailed layer usage inspection
  - task-wise comparison across `boolq`, `gsm8k`, and `rte`

## Notes

- The main comparison uses the `transformers` backend for all methods to keep evaluation fair.
- We do not use `vllm` for the final comparison, since AttnRes currently relies on custom model logic and is not integrated into the same optimized inference path as standard LoRA models.
- AttnRes currently behaves more like a research prototype than a production-ready PEFT method: training, saving/loading, and analysis require more custom code than LoRA.

## Why It Is Still Interesting

Although the current results are still behind LoRA, we think the method is meaningful because it explores a different PEFT direction. Instead of changing weight matrices, AttnRes-PEFT adapts the model by changing how residual information is reused across layers. Our results suggest that this depth-wise routing can already help, especially on reasoning-heavy tasks such as GSM8K.

This direction is also naturally compatible with methods such as LoRA. A hybrid design could let one component change what each layer computes, while the other changes how information is routed across depth. We see this as a promising direction for future work.
