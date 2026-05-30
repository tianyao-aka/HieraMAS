# HieraMAS

## Layout

- `HieraMAS/`: learnable multi-agent graph, LLM routing, prompts, GNN selector/classifier, reward tracking, and utilities used by the method.
- `experiments_v2/humaneval/`: HumanEval stage 1 training, stage 2 data generation/classifier/inference scripts.
- `experiments_v2/math/`: MATH stage 1 training, stage 2 data generation/classifier/inference scripts.
- `experiments_v2/mmlu/`: MMLU stage 1 training, stage 2 data generation/classifier/inference scripts.
- `experiments_v2/*/bash/`: canonical commands for each dataset.
- `Datasets/`: raw dataset exports and JSONL splits used by the experiment scripts.

## Setup

Use Python 3.10.

```bash
conda create -n hieramas python=3.10
conda activate hieramas
pip install -r requirements.txt
cp .env.example .env
```

This release uses OpenRouter as the only LLM API backend. Direct OpenAI and
Together backends are not used. Create an OpenRouter API key, add it to `.env`,
and do not commit `.env`. OpenRouter's authentication docs are at
`https://openrouter.ai/docs/api-keys`.

```bash
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```


## Data

Only raw dataset exports are included under `Datasets/`. See
`Datasets/SOURCE.md` for source details.

Raw exports:

```text
Datasets/MMLU/mmlu_redux_raw.jsonl
Datasets/MATH/math_raw_train.jsonl
Datasets/MATH/math_raw_test.jsonl
Datasets/humaneval/humaneval_raw.jsonl
```

Before running experiments, split the raw JSONL files into train/validation/test
files yourself. The scripts expect JSONL records with the same fields as the raw
exports, and you can either write split files to the default paths below or pass
custom paths with `--dataset_json`, `--valid_dataset_json`, and `--test_file`.

Default split paths should look like:

```text
Datasets/MMLU/train.jsonl
Datasets/MMLU/valid.jsonl
Datasets/MMLU/test.jsonl
Datasets/MMLU/test_unseen.jsonl
Datasets/MATH/math_train.jsonl
Datasets/MATH/math_valid.jsonl
Datasets/MATH/math_test.jsonl
Datasets/humaneval/splits/train.jsonl
Datasets/humaneval/splits/valid.jsonl
Datasets/humaneval/splits/test.jsonl
```

Generated checkpoints and results are written under `result_v2/` and are ignored by git.

## Running Experiments

Run from the repository root.

MMLU:

```bash
bash experiments_v2/mmlu/bash/stage1_training.sh
bash experiments_v2/mmlu/bash/stage2_datagen.sh
bash experiments_v2/mmlu/bash/stage2_classifier.sh
bash experiments_v2/mmlu/bash/stage2_inference.sh
```

MATH:

```bash
bash experiments_v2/math/bash/stage1_train.sh
bash experiments_v2/math/bash/stage2_datagen.sh
bash experiments_v2/math/bash/stage2_classifier.sh
bash experiments_v2/math/bash/stage2_inference.sh
```

HumanEval:

```bash
bash experiments_v2/humaneval/bash/stage1_train.sh
bash experiments_v2/humaneval/bash/stage2_datagen.sh
bash experiments_v2/humaneval/bash/stage2_classifier.sh
bash experiments_v2/humaneval/bash/stage2_inference.sh
```


## Notes

- The `openai` Python package is kept only as an OpenRouter-compatible client; the code points it at `https://openrouter.ai/api/v1`.
- Stage 2 inference expects the stage 1 and stage 2 checkpoint paths produced by the earlier scripts.


This code extends [GDesigner](https://github.com/yanweiyue/GDesigner). We thank the original GDesigner authors for releasing their codebase.
