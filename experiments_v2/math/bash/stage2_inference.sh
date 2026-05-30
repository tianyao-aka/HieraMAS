#!/bin/bash
python experiments_v2/math/run_stage2_inference.py \
    --stage1_checkpoint result_v2/math/stage1/checkpoint/graph_iter10.pt \
    --stage2_checkpoint result_v2/math/stage2/classifier/stage2_classifier_best.pt \
    --test_file Datasets/MATH/math_test.jsonl \
    --output_file result_v2/math/stage2/inference/test_results.json \
    --num_random_graphs 100 \
    --top_k_graphs 5 \
    --num_votes 1 \
    --temperature 0.8 \
    --topK_roles_removed 0 \
    --parallel_batch_size 6 \
    --seed 1
