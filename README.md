# HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning

## Overview

This is the official repo for "HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning".

## Structure
The code includes four base models employed in the paper:
* [BLIP2](https://github.com/salesforce/LAVIS/tree/main)
* [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)
* [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/tree/main)
* [LLaVA](https://github.com/haotian-liu/LLaVA#llava-weights)

## Training & Inference 
We include the training and evaluation code for BLIP2 and MiniGPT4 but not for LLaVA and mPLUG-Owl.

Every base model follows its original released repo's setup environment requirement. You can folow the requirement.txt/.yaml files inside each folder.

## Evaluation
eval_metrics.py includes both evaluation code for VQA v2 and conventional metrics (BLEU, CIDER, ROUGE, and METEOR).

## Data Generation
Our Machine-Human-in-the-Loop adapts the [IdealGPT](https://github.com/Hxyou/IdealGPT) framework for generating part of our initial question-answer pairs.


