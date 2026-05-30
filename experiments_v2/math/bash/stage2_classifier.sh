#!/bin/bash

python experiments_v2/math/run_stage2_classifier.py \
    --dataset_file result_v2/math/stage2/math_stage2_dataset.pkl \
    --output_dir result_v2/math/stage2/classifier \
    --gcn_hidden_dim 256 \
    --gcn_output_dim 128 \
    --classifier_hidden_dim 64 \
    --dropout 0.5 \
    --num_epochs 20 \
    --batch_size 64 \
    --lr 5e-4 \
    --weight_decay 1e-5 \
    --patience 8 \
    --train_ratio 0.8 \
    --seed 1
