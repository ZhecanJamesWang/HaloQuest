# HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning

## Overview

Welcome to the code repository of our ECCV 2024 paper, [**HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning**](https://arxiv.org/abs/2407.15680). This repository contains code, models, evaluation metrics, and information related to our dataset and research paper.

The data repository can be accessed [here](https://github.com/google/haloquest).

## Structure

- `base-models`: Implementation code for four base models.
- `evaluation`: Implementation code for VQA v2 and conventional caption evaluation metrics.
- `examples`: Examples of Haloquest data.


## Base Models

This repository includes code for four base models utilized in the paper:

- [BLIP2](https://github.com/salesforce/LAVIS/tree/main)
- [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/tree/main)
- [LLaVA](https://github.com/haotian-liu/LLaVA#llava-weights)

## Training & Inference

Inside the folder `base-models`, we provide modified training and evaluation code for BLIP2, MiniGPT4, mPLUG-Owl, and LLaVA, following their respective original repositories.

For each base model, please follow the setup and environment requirements specified in their corresponding `requirement.txt/.yaml` or `.toml` files within their respective folders.

## Evaluation

Inside the folder `evaluation`, the `eval_metrics.py` file contains evaluation code for both VQA v2 and conventional metrics such as BLEU, CIDER, ROUGE, and METEOR. 


## HaloQuest Data

For your reference, we provide some examples of HaloQuest data here.

![Examples](examples/examples.png)


Thank you for your interest and patience. Please stay tuned for updates!
