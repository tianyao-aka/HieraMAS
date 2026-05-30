#!/bin/bash

python experiments_v2/mmlu/run_stage2_classifier.py \
    --dataset_file result_v2/mmlu/stage2/mmlu_stage2_dataset.pkl \
    --output_dir result_v2/mmlu/stage2/classifier \
    --gcn_hidden_dim 256 \
    --gcn_output_dim 128 \
    --classifier_hidden_dim 64 \
    --dropout 0.05 \
    --num_epochs 20 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 2e-5 \
    --patience 5 \
    --train_ratio 0.8 \
    --seed 1
