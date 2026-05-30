#!/bin/bash
python experiments_v2/humaneval/run_stage2_inference_humaneval.py \
    --stage1_checkpoint result_v2/humaneval/stage1/checkpoint/bestModel/best_valid_model.pt \
    --stage2_checkpoint result_v2/humaneval/stage2/classifier/stage2_classifier_best.pt \
    --test_file Datasets/humaneval/splits/test.jsonl \
    --output_file result_v2/humaneval/stage2/inference/test_results.json \
    --domain humaneval \
    --decision_method FinalWriteCode \
    --num_random_graphs 200 \
    --top_k_graphs 5 \
    --num_votes 1 \
    --temperature 0.8 \
    --topK_roles_removed 0 \
    --parallel_batch_size 6 \
    --checkpoint_interval 1 \
    --seed 1


python experiments_v2/humaneval/run_stage2_inference_humaneval.py \
    --stage1_checkpoint result_v2/humaneval/stage1/checkpoint/bestModel/best_valid_model.pt \
    --stage2_checkpoint result_v2/humaneval/stage2/classifier/stage2_classifier_best.pt \
    --test_file Datasets/humaneval/splits/test.jsonl \
    --output_file result_v2/humaneval/stage2/inference/qwen3_80b/test_results.json \
    --domain humaneval \
    --decision_method FinalWriteCode \
    --num_random_graphs 200 \
    --top_k_graphs 5 \
    --num_votes 1 \
    --temperature 0.8 \
    --topK_roles_removed 0 \
    --parallel_batch_size 6 \
    --checkpoint_interval 1 \
    --use_qwen3_80b \
    --seed 1