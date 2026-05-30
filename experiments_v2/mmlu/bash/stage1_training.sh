#!/bin/bash
python experiments_v2/mmlu/run_mmlu_learnable_graph.py \
    --batch_size 4 \
    --num_iterations 12 \
    --num_rounds 1 \
    --share_selector \
    --use_random_graph_pool \
    --num_graph_candidates 100 \
    --checkpoint_interval 1 \
    --eval_steps 3 \
    --initial_temp 10.0 \
    --lr_start 5e-3 \
    --lr_end 2e-3 \
    --enable_skip_learning \
    --skip_coef 0.0 \
    --result_file result_v2/mmlu/8workers/stage1/stage1_training.json

