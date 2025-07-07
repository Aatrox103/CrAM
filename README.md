# CrAM

This is the official repo of the paper [CrAM: Credibility-Aware Attention Modification in LLMs for Combating Misinformation in RAG](https://arxiv.org/abs/2406.11497) accepted to AAAI 2025. 

## Overview

### Environment Setup
- **Python Version**: 3.8.19
- To install the required dependencies, run the following command:
  ```bash
  pip install -r requirements.txt
  ```

### Results

- All of our results can be found in the following folders:
  - `results`: Contains experimental results without misinformation.
  - `results_gpt_setting`: Includes results under the GPT setting.
  - `results_ideal_setting`: Contains results under the ideal setting.

### Evaluation Code

- You can find example code for evaluation in the `run.sh` script.
- The `nq_1000_bge.json` file is a sampled subset of the NQ dataset, containing retrieved documents alongside our generated misinformation. The `trivia_1000_bge.jsonl` file is similar.

### Core Code

- The core code for modifying attention weights is located in the `utils/re_weighting.py` file, specifically in the [`Re_Weighting_Strategy` class](./utils/re_weighting.py#L22). Since the transformers library uses the attention_mask multiplied by the final attention_score to achieve the masking effect, our main idea is to use a hook function to modify the attention_mask in order to adjust the attention_score.
- The core code for calculating the impact of each head on the final result is in the same file, in the [`Find_Best_Heads(Re_Weighting_Strategy)` class](./utils/re_weighting.py#L141).

### Influential Heads

- We provide the data used to identify influential heads in the `datasets` directory.

- For example, in `datasets/nq/llama3/heads_scores_mean.json`, you will find entries such as:

  ```json
  [
      0.5078751373291016,  // IE value
      [
          30,  // layer
          14   // head index
      ]
  ]
  ```

  - This indicates that the mean IE value for NQ of head 14 in layer 30 of LLama3 is 0.5078751373291016.

  - The entries are sorted by the IE value, so the first k entries can be considered the top-k influential heads.
