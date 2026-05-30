#!/bin/bash
# Stage 1: Train LLM selector for MATH dataset

python experiments_v2/math/run_math_learnable_graph.py \
    --dataset_json Datasets/MATH/math_train.jsonl \
    --valid_dataset_json Datasets/MATH/math_valid.jsonl \
    --result_file result_v2/math/stage1/stage1_training.json \
    --batch_size 6 \
    --num_iterations 12 \
    --num_rounds 1 \
    --share_selector \
    --use_random_graph_pool \
    --num_graph_candidates 100 \
    --checkpoint_interval 2 \
    --eval_steps 2 \
    --initial_temp 10.0 \
    --lr_start 2e-3 \
    --lr_end 2e-3 \
    --enable_skip_learning \
    --skip_coef 0.0
