#!/bin/bash
python experiments_v2/humaneval/run_stage2_classifier_humaneval.py \
    --dataset_file "result_v2/humaneval/stage2/humaneval_stage2_dataset.pkl" \
    --output_dir "result_v2/humaneval/stage2/classifier" \
    --num_epochs 20 \
    --batch_size 32 \
    --lr 5e-4 \
    --patience 10 \
    --train_ratio 0.8 \
    --gcn_hidden_dim 256 \
    --gcn_output_dim 128 \
    --classifier_hidden_dim 64 \
    --dropout 0.5
