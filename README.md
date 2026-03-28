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

## Development Plan

| Task | Content | Owner |
|---|---|---|
| Algorithm | Design AttnRes as a PEFT method and implement the core algorithm modules. | Weiguo |
| Data | Download datasets and preprocess them into the required format for training and evaluation. | Shaowei |
| Trainer | Build the training pipeline, run fine-tuning experiments, and save checkpoints and training logs. | Shaowei |
| Evaluator | Evaluate models before and after training, and save metrics, inference speed, memory usage, and other detailed results. | Shaowei |

## Core IDEA

AttnRes-PEFT turns each Transformer layer into a small decision point: instead of only trusting the current layer output, the model can also "look back" across earlier hidden states and borrow the most useful representation depth-wise.

At every layer, the adapter does three things:

1. Collect previous hidden states from earlier depths.
2. Compute depth attention weights over those states with a learned query.
3. Build a weighted residual correction and add it to the frozen backbone output.

In compact form:

$$
\Delta h_t^{(\ell)} = \sum_{k < \ell} \alpha_{k,t}^{(\ell)} h_t^{(k)}, \qquad
h_{\text{out}}^{(\ell)} = h_{\text{base}}^{(\ell)} + g^{(\ell)} \cdot \Delta h_t^{(\ell)}
$$

where $\alpha$ is a softmax over depth and $g^{(\ell)}$ is a learnable scalar gate for layer $\ell$.

### Why this is PEFT-friendly

- The backbone model is frozen.
- Only lightweight adapter parameters are trainable (query, norms, and gate per layer).
- This keeps memory and trainable parameter count low while still adding expressive adaptation capacity.

### Safe and controllable behavior

The gate is the key control knob:

- If $g=0$, the adapter contribution is exactly zero, so the model behaves like the original backbone.
- Starting from `gate_init=0.0` gives a "do-no-harm" initialization: training begins from base-model behavior and learns to add residual depth information only when useful.
- Increasing learned gate values allows stronger AttnRes influence.

The wrapper also supports practical control modes for analysis:

- `lookback` limits depth attention to recent layers for better efficiency.
- You can route generation/forward through strict base behavior when adapters are effectively disabled, making parity checks and ablations straightforward.

In short, this adapter keeps the spirit of Attention Residuals, but packages it into a clean, switchable, and parameter-efficient fine-tuning mechanism you can compare fairly against LoRA.
