#!/bin/bash

# Start fresh
python experiments_v2/mmlu/run_stage2_datagen.py \
    --stage1_checkpoint result_v2/mmlu/stage1/checkpoint/bestModel/best_valid_model.pt \
    --dataset_json Datasets/MMLU/train.jsonl \
    --output_file result_v2/mmlu/stage2/mmlu_stage2_dataset.pkl \
    --batch_size 8 \
    --num_graphs_per_task 5 \
    --graph_min_density 0.3 \
    --graph_max_density 0.7 \
    --top_k_positive 2 \
    --num_rounds 1 \
    --share_selector \
    --enable_skip_learning \
    --checkpoint_interval 1
