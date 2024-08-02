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

## Leaderboard


<table>
  <thead>
    <tr>
      <th>Model (#Param)</th>
      <th>Rank</th>
      <th>Overall</th>
      <th>Generated</th>
      <th></th>
      <th>Real</th>
      <th></th>
      <th>False Premise</th>
      <th></th>
      <th>Visually Challenging</th>
      <th></th>
      <th>Insufficient Context</th>
      <th></th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Human Eval</th>
      <th>Auto-Eval</th>
      <th>Human Eval</th>
      <th>Auto-Eval</th>
      <th>Human Eval</th>
      <th>Auto-Eval</th>
      <th>Human Eval</th>
      <th>Auto-Eval</th>
      <th>Human Eval</th>
      <th>Auto-Eval</th>
      <th>Human Eval</th>
      <th>Auto-Eval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gemini 1.5 Pro</td>
      <td>1</td>
      <td>76.1</td>
      <td>77.9</td>
      <td>74.7</td>
      <td>78.3</td>
      <td>78.7</td>
      <td>77.2</td>
      <td>80.4</td>
      <td>83.7</td>
      <td>57.3</td>
      <td>56.3</td>
      <td>91</td>
      <td>92.5</td>
    </tr>
    <tr>
      <td>GPT-4o</td>
      <td>2</td>
      <td>68.1</td>
      <td>63.2</td>
      <td>68.8</td>
      <td>63.8</td>
      <td>66.9</td>
      <td>62.2</td>
      <td>68.5</td>
      <td>65.2</td>
      <td>58.3</td>
      <td>55.2</td>
      <td>80.6</td>
      <td>68.7</td>
    </tr>
    <tr>
      <td>GPT-4</td>
      <td>3</td>
      <td>62.9</td>
      <td>61.2</td>
      <td>64.3</td>
      <td>61.1</td>
      <td>60.6</td>
      <td>61.4</td>
      <td>64.7</td>
      <td>63</td>
      <td>46.9</td>
      <td>44.8</td>
      <td>80.6</td>
      <td>79.1</td>
    </tr>
    <tr>
      <td>BEiT-3 (0.7B)</td>
      <td>4</td>
      <td>35.9</td>
      <td>40</td>
      <td>41.2</td>
      <td>44.3</td>
      <td>26.3</td>
      <td>32.3</td>
      <td>24.1</td>
      <td>28.4</td>
      <td>36.6</td>
      <td>36.1</td>
      <td>9.1</td>
      <td>10.7</td>
    </tr>
    <tr>
      <td>InstructBLIP (12B)</td>
      <td>5</td>
      <td>25.5</td>
      <td>28.5</td>
      <td>28.4</td>
      <td>31.5</td>
      <td>20.3</td>
      <td>23</td>
      <td>28.4</td>
      <td>32</td>
      <td>33.3</td>
      <td>33.9</td>
      <td>6.6</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>InstructBLIP (8B)</td>
      <td>6</td>
      <td>25</td>
      <td>27.3</td>
      <td>28.4</td>
      <td>29.7</td>
      <td>18.9</td>
      <td>23</td>
      <td>28.4</td>
      <td>32</td>
      <td>6.6</td>
      <td>11.6</td>
      <td>33.3</td>
      <td>33.9</td>
    </tr>
    <tr>
      <td>BLIP2 (12B)</td>
      <td>7</td>
      <td>21.1</td>
      <td>22.5</td>
      <td>24.8</td>
      <td>26.1</td>
      <td>14.29</td>
      <td>16.1</td>
      <td>16.8</td>
      <td>19.5</td>
      <td>35.5</td>
      <td>32.8</td>
      <td>9.9</td>
      <td>14.9</td>
    </tr>
    <tr>
      <td>MiniGPT4 (13B)</td>
      <td>8</td>
      <td>18.7</td>
      <td>25.2</td>
      <td>18.2</td>
      <td>24</td>
      <td>18.9</td>
      <td>27.2</td>
      <td>16.2</td>
      <td>21.5</td>
      <td>10.4</td>
      <td>13.7</td>
      <td>36.4</td>
      <td>51.2</td>
    </tr>
    <tr>
      <td>MiniGPT4 (7B)</td>
      <td>9</td>
      <td>18.6</td>
      <td>19.1</td>
      <td>18.1</td>
      <td>19.4</td>
      <td>18</td>
      <td>18.4</td>
      <td>13.2</td>
      <td>13.2</td>
      <td>26.5</td>
      <td>27.3</td>
      <td>15.7</td>
      <td>16.5</td>
    </tr>
    <tr>
      <td>Open-flamingo (9B)</td>
      <td>10</td>
      <td>13.8</td>
      <td>15</td>
      <td>16.1</td>
      <td>17.1</td>
      <td>9.7</td>
      <td>11.1</td>
      <td>13.2</td>
      <td>13.9</td>
      <td>19.1</td>
      <td>21.3</td>
      <td>7.4</td>
      <td>8.3</td>
    </tr>
    <tr>
      <td>LLaVA (13B)</td>
      <td>11</td>
      <td>10.9</td>
      <td>10.9</td>
      <td>12.3</td>
      <td>12.8</td>
      <td>8.2</td>
      <td>7.4</td>
      <td>2.3</td>
      <td>1.7</td>
      <td>30.6</td>
      <td>31.2</td>
      <td>2.5</td>
      <td>3.3</td>
    </tr>
    <tr>
      <td>BLIP2 (8B)</td>
      <td>12</td>
      <td>10.9</td>
      <td>11.8</td>
      <td>11.5</td>
      <td>11.8</td>
      <td>9.7</td>
      <td>12</td>
      <td>5</td>
      <td>4.6</td>
      <td>26

