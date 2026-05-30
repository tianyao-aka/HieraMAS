#!/bin/bash

python experiments_v2/mmlu/run_stage2_inference_baseline.py \
    --test_file Datasets/MMLU/test.jsonl \
    --output_file result_v2/mmlu/baseline/random_graph/gpt5mini/test_results.json \
    --random_graph \
    --decision_llm gpt-5-mini \
    --parallel_batch_size 6 \
    --seed 1


python experiments_v2/mmlu/run_stage2_inference_baseline.py \
    --test_file Datasets/MMLU/test.jsonl \
    --output_file result_v2/mmlu/baseline/random_graph/qwen3_80b/test_results.json \
    --random_graph \
    --decision_llm qwen3-80b \
    --parallel_batch_size 6 \
    --seed 1
