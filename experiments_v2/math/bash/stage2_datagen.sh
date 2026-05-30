#!/bin/bash
python experiments_v2/math/run_stage2_datagen.py \
    --stage1_checkpoint result_v2/math/stage1/checkpoint/bestModel/best_valid_model.pt \
    --dataset_json Datasets/MATH/math_train.jsonl \
    --output_file result_v2/math/stage2/math_stage2_dataset.pkl \
    --num_graphs_per_task 5 \
    --graph_min_density 0.3 \
    --graph_max_density 1.0 \
    --top_k_positive 2 \
    --batch_size 6 \
    --domain gsm8k \
    --seed 1
