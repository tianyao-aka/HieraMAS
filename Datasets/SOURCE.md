# Dataset Sources

The JSONL files in this directory are raw exports from Hugging Face datasets.
Train/validation/test splits are intentionally not committed; create them
locally before running experiments.

## MMLU-Redux

- Source: `edinburgh-dawg/mmlu-redux-2.0`
- Raw export: `MMLU/mmlu_redux_raw.jsonl`
- Upstream split: `test` for each subject config
- Split note: MMLU-Redux 2.0 exposes test rows per subject. If you need
  train/validation/test files for the provided scripts, partition the combined
  raw export locally.

## MATH

- Source: `jeggers/competition_math`, config `original`
- Raw exports: `MATH/math_raw_train.jsonl`, `MATH/math_raw_test.jsonl`
- Upstream splits: `train`, `test`
- Split note: Use the upstream train split to create your local train and
  validation files. Use the upstream test split for test evaluation.

## HumanEval

- Source: `openai/openai_humaneval`, config `openai_humaneval`
- Raw export: `humaneval/humaneval_raw.jsonl`
- Upstream split: `test`
- Split note: HumanEval exposes a single test split. If you need
  train/validation/test files for the provided scripts, partition the raw export
  locally.
