#!/bin/bash

# Base method
# python experiments_v2/baselines/run_mmlu_baseline.py \
#     --method base \
#     --model gpt-5-mini \
#     --dataset_path Datasets/MMLU/test_unseen.jsonl \
#     --output_dir result_v2/mmlu/unseen/base/

# SC+CoT method
python experiments_v2/baselines/run_mmlu_baseline.py \
    --method cot_sc \
    --model qwen3-80b \
    --dataset_path Datasets/MMLU/test_unseen.jsonl \
    --output_dir result_v2/mmlu/unseen/qwen3_80b/cot_sc/
