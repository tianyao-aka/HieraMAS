#!/bin/bash

python experiments_v2/mmlu/run_stage2_inference.py \
    --stage1_checkpoint result_v2/mmlu/stage1/checkpoint/bestModel/best_valid_model.pt \
    --stage2_checkpoint result_v2/mmlu/stage2/classifier/stage2_classifier_best.pt \
    --test_file Datasets/MMLU/test.jsonl \
    --output_file result_v2/mmlu/stage2/inference/random_graph/test_results_random_graph.json \
    --random_graph \
    --temperature 0.8 \
    --parallel_batch_size 6 \
    --seed 1
