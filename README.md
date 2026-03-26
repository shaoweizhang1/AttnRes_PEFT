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
