"""
MATH Dataset Download and Processing

Downloads the MATH dataset from Hendrycks et al. (NeurIPS 2021):
"Measuring Mathematical Problem Solving with the MATH Dataset"

Reference:
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart,
Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem
solving with the MATH dataset. NeurIPS Datasets and Benchmarks 2021.
https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html

Dataset source: HuggingFace - qwedsacf/competition_math
"""

import os
import json
import importlib
from pathlib import Path
from typing import List, Dict, Any

try:
    # Use importlib to avoid conflict with local 'datasets' directory
    hf_datasets = importlib.import_module("datasets")
    load_dataset = hf_datasets.load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    raise


def download_math_dataset(output_dir: str = None, split: str = "test") -> List[Dict[str, Any]]:
    """
    Download the MATH dataset from HuggingFace and save as JSONL.

    Args:
        output_dir: Directory to save the dataset. Defaults to datasets/MATH/
        split: Which split to download ('train' or 'test')

    Returns:
        List of processed data records
    """
    if output_dir is None:
        this_file_path = Path(__file__).parent
        output_dir = this_file_path / "MATH"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MATH dataset from HuggingFace (split: {split})...")
    print("Source: qwedsacf/competition_math")

    # Load from HuggingFace - using the correct dataset name from the README
    dataset = load_dataset("qwedsacf/competition_math", split=split, trust_remote_code=True)

    print(f"Downloaded {len(dataset)} problems")

    all_records = []

    for idx, item in enumerate(dataset):
        record = {
            "task": item.get("problem", ""),
            "solution": item.get("solution", ""),
            "answer": extract_answer(item.get("solution", "")),
            "level": item.get("level", ""),
            "type": item.get("type", ""),
        }
        all_records.append(record)

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} problems")

    # Save to JSONL
    output_file = output_dir / "math.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\nTotal problems: {len(all_records)}")
    print(f"Saved to {output_file}")

    # Print statistics
    types = {}
    levels = {}
    for r in all_records:
        t = r.get("type", "unknown")
        l = r.get("level", "unknown")
        types[t] = types.get(t, 0) + 1
        levels[l] = levels.get(l, 0) + 1

    print("\nProblem types:")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")

    print("\nDifficulty levels:")
    for l, count in sorted(levels.items()):
        print(f"  {l}: {count}")

    return all_records


def extract_answer(solution: str) -> str:
    """
    Extract the final answer from a MATH solution.
    The answer is typically in \\boxed{...} format.

    Args:
        solution: The solution string

    Returns:
        Extracted answer string
    """
    if "\\boxed" not in solution:
        return ""

    # Find the last \\boxed{...}
    idx = solution.rfind("\\boxed")
    if idx == -1:
        return ""

    # Extract content within braces
    remaining = solution[idx + len("\\boxed"):]
    if not remaining or remaining[0] != '{':
        # Handle \\boxed without braces (e.g., \\boxed5)
        if remaining:
            return remaining[0]
        return ""

    # Match balanced braces
    depth = 0
    start = 0
    for i, char in enumerate(remaining):
        if char == '{':
            if depth == 0:
                start = i + 1
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return remaining[start:i]

    return ""


def math_data_process(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process MATH dataset for use in experiments.
    Similar to gsm_data_process for GSM8K.

    Args:
        dataset: List of MATH dataset records

    Returns:
        List of processed records with 'task', 'step', 'answer' fields
    """
    list_data_dict = []
    for data in dataset:
        item = {
            "task": data.get("task", data.get("problem", "")),
            "step": data.get("solution", ""),
            "answer": data.get("answer", extract_answer(data.get("solution", ""))),
            "level": data.get("level", ""),
            "type": data.get("type", ""),
        }
        list_data_dict.append(item)
    return list_data_dict


def math_get_predict(pred_str: str) -> str:
    """
    Extract prediction from model output.
    Similar to gsm_get_predict for GSM8K.

    Args:
        pred_str: Model prediction string

    Returns:
        Extracted answer
    """
    import re

    # Try "The answer is" pattern
    if 'The answer is ' in pred_str:
        pred = pred_str.split('The answer is ')[-1].strip()
        # Clean up trailing punctuation
        pred = pred.rstrip('.').strip()
        return pred
    elif 'the answer is ' in pred_str:
        pred = pred_str.split('the answer is ')[-1].strip()
        pred = pred.rstrip('.').strip()
        return pred

    # Try boxed pattern
    if 'boxed' in pred_str:
        return extract_answer(pred_str)

    # Fallback: try to find the last number or expression
    # This is a simple fallback - MATH answers can be complex expressions
    pattern = r'-?\d*\.?\d+'
    matches = re.findall(pattern, pred_str)
    if matches:
        return matches[-1]
    return pred_str.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and process MATH dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to download (default: test)"
    )

    args = parser.parse_args()

    records = download_math_dataset(split=args.split)
    print(f"\nDone! Downloaded {len(records)} problems.")
