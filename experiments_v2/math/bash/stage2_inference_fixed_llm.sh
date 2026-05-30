#!/bin/bash

python experiments_v2/math/run_stage2_inference.py \
    --stage1_checkpoint result_v2/math/stage1/checkpoint/graph_iter10.pt \
    --stage2_checkpoint result_v2/math/stage2/classifier/stage2_classifier_best.pt \
    --test_file Datasets/MATH/math_test.jsonl \
    --output_file result_v2/math/stage2/inference/fixed_llm/test_results_fixed_llm.json \
    --fixed_llm \
    --num_random_graphs 200 \
    --top_k_graphs 1 \
    --num_votes 1 \
    --temperature 0.8 \
    --parallel_batch_size 6 \
    --seed 1
