# HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning

## Overview

Welcome to the official repository for **HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning**. This repository contains code, models, evaluation metrics, and information related to our dataset and research paper.

## Structure

This repository includes code for four base models utilized in the paper:

- [BLIP2](https://github.com/salesforce/LAVIS/tree/main)
- [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/tree/main)
- [LLaVA](https://github.com/haotian-liu/LLaVA#llava-weights)

## Training & Inference

We provide training and evaluation code for BLIP2, MiniGPT4, mPLUG-Owl and LLaVA, following their respective original repositories.

Each base model follows the setup and environment requirements specified in their corresponding `requirement.txt` or `requirement.yaml` files within their respective folders.

## Evaluation

The `eval_metrics.py` file contains evaluation code for both VQA v2 and conventional metrics such as BLEU, CIDER, ROUGE, and METEOR.

## Automatic Data Generation

We utilize the Machine-Human-in-the-Loop approach, adapting the [IdealGPT](https://github.com/Hxyou/IdealGPT) framework to generate a portion of our initial question-answer pairs.

## Data

While we are actively working on open-sourcing the dataset, this process is not yet complete. However, we aim to have it ready in time for the conference.

Thank you for your interest and patience. Please stay tuned for updates!
