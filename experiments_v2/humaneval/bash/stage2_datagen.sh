#!/bin/bash
python experiments_v2/humaneval/run_stage2_datagen_humaneval.py \
    --stage1_checkpoint "result_v2/humaneval/stage1/checkpoint/bestModel/best_valid_model.pt" \
    --dataset_json Datasets/humaneval/splits/train.jsonl \
    --output_file "result_v2/humaneval/stage2/humaneval_stage2_dataset.pkl" \
    --num_graphs_per_task 5 \
    --graph_min_density 0.3 \
    --graph_max_density 0.8 \
    --top_k_positive 2 \
    --batch_size 6 \
    --checkpoint_interval 1 \
    --domain humaneval \
    --decision_method FinalWriteCode \
    --decision_llm "gpt-5-mini" \
    --openrouter_key_variant 1 \
    --share_selector
