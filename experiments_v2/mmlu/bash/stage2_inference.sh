#!/bin/bash

python experiments_v2/mmlu/run_stage2_inference.py \
    --stage1_checkpoint result_v2/mmlu/stage1/checkpoint/bestModel/best_valid_model.pt \
    --stage2_checkpoint result_v2/mmlu/stage2/classifier/stage2_classifier_best.pt \
    --test_file Datasets/MMLU/test.jsonl \
    --output_file result_v2/mmlu/stage2/inference/test_results.json \
    --num_random_graphs 200 \
    --top_k_graphs 5 \
    --num_votes 1 \
    --temperature 0.8 \
    --topK_roles_removed 2 \
    --parallel_batch_size 6 \
    --seed 1


python experiments_v2/mmlu/run_stage2_inference.py \
    --stage1_checkpoint result_v2/mmlu/stage1/checkpoint/bestModel/best_valid_model.pt \
    --stage2_checkpoint result_v2/mmlu/stage2/classifier/stage2_classifier_best.pt \
    --test_file Datasets/MMLU/test.jsonl \
    --output_file result_v2/mmlu/stage2/inference/K_50/test_results.json \
    --num_random_graphs 50 \
    --top_k_graphs 5 \
    --num_votes 1 \
    --temperature 0.8 \
    --topK_roles_removed 2 \
    --parallel_batch_size 6 \
    --seed 1

python experiments_v2/mmlu/run_stage2_inference.py \
    --stage1_checkpoint result_v2/mmlu/stage1/checkpoint/bestModel/best_valid_model.pt \
    --stage2_checkpoint result_v2/mmlu/stage2/classifier/stage2_classifier_best.pt \
    --test_file Datasets/MMLU/test.jsonl \
    --output_file result_v2/mmlu/stage2/inference/K_500/test_results.json \
    --num_random_graphs 500 \
    --top_k_graphs 5 \
    --num_votes 1 \
    --temperature 0.8 \
    --topK_roles_removed 2 \
    --parallel_batch_size 6 \
    --openrouter_key_variant 2 \
    --seed 1