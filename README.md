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

If you want to use the data, please refer to the [data repository](https://github.com/google/haloquest).

## Leaderboard

| Model (#Param)     | Rank | Overall | Generated         | Real             | False Premise      | Visually Challenging | Insufficient Context |
|--------------------|------|---------|-------------------|------------------|--------------------|----------------------|----------------------|
|                    |      | Human Eval | Auto-Eval | Human Eval | Auto-Eval | Human Eval | Auto-Eval | Human Eval | Auto-Eval | Human Eval | Auto-Eval | Human Eval | Auto-Eval | Human Eval | Auto-Eval |
| Gemini 1.5 Pro     | 1    | 76.1    | 77.9    | 74.7    | 78.3    | 78.7    | 77.2    | 80.4    | 83.7    | 57.3    | 56.3    | 91      | 92.5    |
| GPT-4o             | 2    | 68.1    | 63.2    | 68.8    | 63.8    | 66.9    | 62.2    | 68.5    | 65.2    | 58.3    | 55.2    | 80.6    | 68.7    |
| GPT-4              | 3    | 62.9    | 61.2    | 64.3    | 61.1    | 60.6    | 61.4    | 64.7    | 63      | 46.9    | 44.8    | 80.6    | 79.1    |
| BEiT-3 (0.7B)      | 4    | 35.9    | 40      | 41.2    | 44.3    | 26.3    | 32.3    | 24.1    | 28.4    | 36.6    | 36.1    | 9.1     | 10.7    |
| InstructBLIP (12B) | 5    | 25.5    | 28.5    | 28.4    | 31.5    | 20.3    | 23      | 28.4    | 32      | 33.3    | 33.9    | 6.6     | 11.6    |
| InstructBLIP (8B)  | 6    | 25      | 27.3    | 28.4    | 29.7    | 18.9    | 23      | 28.4    | 32      | 6.6     | 11.6    | 33.3    | 33.9    |
| BLIP2 (12B)        | 7    | 21.1    | 22.5    | 24.8    | 26.1    | 14.29   | 16.1    | 16.8    | 19.5    | 35.5    | 32.8    | 9.9     | 14.9    |
| MiniGPT4 (13B)     | 8    | 18.7    | 25.2    | 18.2    | 24      | 18.9    | 27.2    | 16.2    | 21.5    | 10.4    | 13.7    | 36.4    | 51.2    |
| MiniGPT4 (7B)      | 9    | 18.6    | 19.1    | 18.1    | 19.4    | 18      | 18.4    | 13.2    | 13.2    | 26.5    | 27.3    | 15.7    | 16.5    |
| Open-flamingo (9B) | 10   | 13.8    | 15      | 16.1    | 17.1    | 9.7     | 11.1    | 13.2    | 13.9    | 19.1    | 21.3    | 7.4     | 8.3     |
| LLaVA (13B)        | 11   | 10.9    | 10.9    | 12.3    | 12.8    | 8.2     | 7.4     | 2.3     | 1.7     | 30.6    | 31.2    | 2.5     | 3.3     |
| BLIP2 (8B)         | 12   | 10.9    | 11.8    | 11.5    | 11.8    | 9.7     | 12      | 5       | 4.6     | 26.8    | 26.8    | 1.7     | 6.6     |
| mPLUG-Owl1 (7B)    | 13   | 9.7     | 8.7     | 11.3    | 10.2    | 6.9     | 6       | 1       | 0.3     | 29      | 26.8    | 2.5     | 2.5     |
| mPLUG-Owl2 (7B)    | 14   | 9.2     | 10.4    | 11      | 11.3    | 6       | 8.8     | 0.8     | 3.3     | 28.4    | 27.9    | 0.8     | 3.3     |
| OFA (1B)           | 15   | 8.7     | 10.2    | 9.7     | 11.3    | 6.9     | 8.3     | 5       | 6.3     | 19.7    | 20.2    | 1.7     | 5       |
| Open-flamingo (3B) | 16   | 6.9     | 8.2     | 7.4     | 8.7     | 6       | 7.4     | 0.7     | 1.3     | 19.1    | 21.3    | 4.1     | 5.8     |


## Contributions

[Zhecan Wang](https://www.zhecanwang.com/)\*, [Garrett Bingham](https://garrettbingham.com/)\*, [Adams Wei Yu](https://adamsyu.github.io/), [Quoc V. Le](https://research.google/people/quoc-v-le/?&type=google), [Thang Luong](https://nlp.stanford.edu/~lmthang/), [Golnaz Ghiasi](https://research.google/people/golnaz-ghiasi/?&type=google)

(\* ZW and GB are main contributors. ZW did some work while at Google DeepMind.)

## Citing this work

```latex
@inproceedings{wang2024haloquest,
  title={HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning},
  author={Zhecan Wang and Garrett Bingham and Adams Wei Yu and Quoc V. Le and Thang Luong and Golnaz Ghiasi},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```
