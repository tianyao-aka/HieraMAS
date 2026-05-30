#!/bin/bash

python experiments_v2/humaneval/run_stage2_inference_humaneval.py \
    --stage1_checkpoint result_v2/humaneval/stage1/checkpoint/bestModel/best_valid_model.pt \
    --test_file Datasets/humaneval/splits/test.jsonl \
    --output_file result_v2/humaneval/stage2/inference/test_results_random_graph.json \
    --random_graph \
    --temperature 0.8 \
    --parallel_batch_size 6 \
    --seed 1
