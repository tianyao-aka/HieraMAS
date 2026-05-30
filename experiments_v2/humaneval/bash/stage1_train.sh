#!/bin/bash
python experiments_v2/humaneval/run_humaneval_learnable_graph.py \
    --dataset_json Datasets/humaneval/splits/train.jsonl \
    --valid_dataset_json Datasets/humaneval/splits/valid.jsonl \
    --use_random_graph_pool \
    --num_graph_candidates 100 \
    --graph_min_density 0.3 \
    --graph_max_density 0.75 \
    --batch_size 6 \
    --num_iterations 12 \
    --lr_start 2e-3 \
    --lr_end 2e-3 \
    --warmup_ratio 0.5 \
    --entropy_coef 1.0 \
    --initial_temp 10.0 \
    --min_temp 0.8 \
    --checkpoint_interval 2 \
    --eval_steps 2 \
    --share_selector \
    --mode FullConnected \
    --num_rounds 1 \
    --domain humaneval \
    --decision_method FinalWriteCode \
    --openrouter_key_variant 1 \
    --enable_skip_learning \
    --skip_coef 0.0 

