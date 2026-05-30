"""
Shared utilities for baseline experiments.
"""

import re
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union
from collections import Counter
from tenacity import retry, stop_after_attempt, wait_fixed


def majority_vote(answers: List[str]) -> str:
    """
    Return the most common answer from a list of answers.

    Args:
        answers: List of answer strings

    Returns:
        Most common answer (ties broken by first occurrence)
    """
    if not answers:
        return ""
    # Filter out empty strings
    valid_answers = [a for a in answers if a]
    if not valid_answers:
        return ""
    counter = Counter(valid_answers)
    return counter.most_common(1)[0][0]


def extract_mmlu_answer(response: str) -> str:
    """
    Extract A/B/C/D answer from MMLU response.

    Args:
        response: Model response string

    Returns:
        Single letter (A, B, C, or D) or empty string
    """
    if not response:
        return ""

    response = response.strip()

    # First, try to find "answer is" pattern
    ans_pos = response.lower().find("answer is")
    if ans_pos != -1:
        after_answer = response[ans_pos + len("answer is"):].strip()
        after_answer = after_answer.lstrip(":").strip()
        # Remove "Option" prefix if present
        if after_answer.lower().startswith("option"):
            after_answer = after_answer[6:].strip()
        if after_answer and after_answer[0].upper() in "ABCD":
            return after_answer[0].upper()

    # Try to find standalone letter at the beginning
    if response and response[0].upper() in "ABCD":
        # Check if it's a standalone answer (followed by punctuation or space)
        if len(response) == 1 or response[1] in " .),:;\n":
            return response[0].upper()

    # Look for patterns like "A)", "A.", "(A)", etc.
    patterns = [
        r'\b([ABCD])\s*[).]',  # A) or A.
        r'\(([ABCD])\)',       # (A)
        r'\b([ABCD])\b',       # standalone A B C D
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()

    # Fallback: take first letter if it's A-D
    if response and response[0].upper() in "ABCD":
        return response[0].upper()

    return ""


@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
async def extract_mmlu_answer_llm(
    response: str,
    gold_answer: str,
    llm=None,
) -> str:
    """
    Use LLM to determine if response matches gold answer.

    Args:
        response: Model response string
        gold_answer: The correct answer letter (A, B, C, or D)
        llm: Optional LLM instance (will create gpt-4o-mini if None)

    Returns:
        "yes" if response matches gold answer, "no" otherwise
    """
    if llm is None:
        from HieraMAS.llm.llm_registry import LLMRegistry
        llm = LLMRegistry.get("gpt-4o-mini")

    prompt = f"""Given this response to a multiple choice question, determine if the final answer matches "{gold_answer}".

Response:
{response}

Does the response's final answer match "{gold_answer}"? Reply with only "yes" or "no"."""

    messages = [{"role": "user", "content": prompt}]
    result = await llm.agen(messages=messages, temperature=0, num_comps=1)

    # Extract yes/no from result
    result_lower = result.strip().lower()
    if "yes" in result_lower:
        return "yes"
    return "no"


@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
async def verify_answer_with_llm(
    predicted: str,
    gold: str,
    llm=None,
) -> bool:
    """
    Use LLM to verify if predicted answer matches gold answer.

    Args:
        predicted: Predicted answer string
        gold: Gold/correct answer string
        llm: Optional LLM instance (will create gpt-4o-mini if None)

    Returns:
        True if answers are equivalent, False otherwise
    """
    if llm is None:
        from HieraMAS.llm.llm_registry import LLMRegistry
        llm = LLMRegistry.get("gpt-4o-mini")

    prompt = f"""Compare these two answers and determine if they are equivalent.

Gold answer: {gold}
Predicted answer: {predicted}

Are they equivalent? Reply with only "yes" or "no"."""

    messages = [{"role": "user", "content": prompt}]
    result = await llm.agen(messages=messages, temperature=0, num_comps=1)

    return "yes" in result.strip().lower()


@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
async def extract_answer_with_llm(
    response: str,
    dataset: str,
    llm=None,
) -> str:
    """
    Use gpt-4o-mini to extract the final answer from model response.

    Args:
        response: Model response string
        dataset: Dataset type ("mmlu", "gsm8k", "math")
        llm: Optional LLM instance (will create gpt-4o-mini if None)

    Returns:
        Extracted answer string
    """
    if not response:
        return ""

    if llm is None:
        from HieraMAS.llm.llm_registry import LLMRegistry
        llm = LLMRegistry.get("gpt-4o-mini")

    prompts = {
        "mmlu": f"""Extract the final answer (A, B, C, or D) from this response.

Response: {response}

Reply with only the single letter (A, B, C, or D).""",

        "gsm8k": f"""Extract the final numeric answer from this math problem response.

Response: {response}

Reply with only the number (e.g., 42, -3.5, 100). No units or text.""",

        "math": f"""Extract the final answer from this math problem response.

Response: {response}

Reply with only the answer value. If LaTeX, keep it simple (e.g., 42, 3/4, x^2+1).""",
    }

    prompt = prompts.get(dataset, prompts["math"])
    messages = [{"role": "user", "content": prompt}]
    result = await llm.agen(messages=messages, temperature=0, num_comps=1)

    return result.strip()


def extract_gsm8k_answer(response: str) -> str:
    """
    Extract numeric answer from GSM8K response.

    Args:
        response: Model response string

    Returns:
        Numeric string or empty string
    """
    if not response:
        return ""

    pred = ""

    # Try "The answer is" pattern
    if 'The answer is ' in response:
        pred = response.split('The answer is ')[-1].strip()
    elif 'the answer is ' in response:
        pred = response.split('the answer is ')[-1].strip()
    elif 'boxed' in response:
        pred = _extract_boxed(response)
    else:
        # Fallback: find last number in response
        pattern = r'-?\d*\.?\d+'
        matches = re.findall(pattern, response)
        if matches:
            pred = matches[-1]

    # Clean up the prediction
    pred = _clean_number(pred)

    return pred


def extract_math_answer(response: str) -> str:
    """
    Extract answer from MATH dataset response.
    Handles both \\boxed{} notation and "The answer is" pattern.

    Args:
        response: Model response string

    Returns:
        Answer string
    """
    if not response:
        return ""

    # Try boxed notation first (most common in MATH)
    if 'boxed' in response:
        return _extract_boxed(response)

    # Try "The answer is" pattern
    if 'The answer is ' in response:
        pred = response.split('The answer is ')[-1].strip()
        pred = pred.rstrip('.').strip()
        return pred
    elif 'the answer is ' in response:
        pred = response.split('the answer is ')[-1].strip()
        pred = pred.rstrip('.').strip()
        return pred

    # Fallback: try to find the last number or expression
    pattern = r'-?\d*\.?\d+'
    matches = re.findall(pattern, response)
    if matches:
        return matches[-1]

    return response.strip()


def _extract_boxed(text: str) -> str:
    """
    Extract content from \\boxed{...} notation.

    Args:
        text: String containing boxed notation

    Returns:
        Content inside the boxed notation
    """
    if 'boxed' not in text:
        return ""

    # Find the last boxed
    idx = text.rfind('boxed')
    remaining = text[idx + len('boxed'):]

    if not remaining or remaining[0] != '{':
        # Handle \\boxed without braces (e.g., \\boxed5)
        if remaining:
            # Take until whitespace or end
            match = re.match(r'([^\s{}]+)', remaining)
            if match:
                return match.group(1)
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


def _clean_number(pred: str) -> str:
    """
    Clean up a numeric prediction.

    Args:
        pred: Raw prediction string

    Returns:
        Cleaned numeric string
    """
    if not pred:
        return ""

    # Remove trailing punctuation
    pred = pred.rstrip('.,/')

    # Remove commas from numbers
    pred = pred.replace(',', '')

    # Remove units and extra text
    pred = re.sub(r'[a-zA-Z]+.*$', '', pred).strip()

    # If still contains boxed, extract it
    if 'boxed' in pred:
        pred = _extract_boxed(pred)

    # Try to extract just the number
    if pred:
        matches = re.findall(r'-?\d*\.?\d+', pred)
        if matches:
            pred = matches[-1]

    return pred


def extract_code(response: str) -> str:
    """
    Extract Python code from response.

    Args:
        response: Model response string

    Returns:
        Extracted Python code
    """
    if not response:
        return ""

    # Try to find code block
    code_block_pattern = r'```(?:python)?\s*\n?(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        # Return the last code block (usually the final implementation)
        return matches[-1].strip()

    # If no code block, check if the response is just code
    # (starts with def, class, import, or from)
    stripped = response.strip()
    if stripped.startswith(('def ', 'class ', 'import ', 'from ')):
        return stripped

    # Try to find def statement
    def_match = re.search(r'(def\s+\w+.*?)(?=\n\n|\Z)', response, re.DOTALL)
    if def_match:
        return def_match.group(1).strip()

    return response.strip()


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    dataset_name: str,
    model_name: str,
    method: str,
    summary: Dict[str, Any] = None,
) -> str:
    """
    Save experiment results to JSON file.

    Args:
        results: List of per-sample results
        output_dir: Directory to save results
        dataset_name: Name of the dataset (mmlu, gsm8k, math, humaneval)
        model_name: Name of the model
        method: Baseline method used
        summary: Optional summary statistics

    Returns:
        Path to the saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{dataset_name}_{model_name}_{method}_{timestamp}.json"
    output_path = output_dir / filename

    output_data = {
        "config": {
            "dataset": dataset_name,
            "model": model_name,
            "method": method,
            "timestamp": timestamp,
        },
        "summary": summary or {},
        "results": results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return str(output_path)


def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute accuracy from results.

    Args:
        results: List of results with 'is_correct' field

    Returns:
        Summary dictionary with accuracy metrics
    """
    if not results:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for r in results if r.get('is_correct', False))
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


# Method configurations
METHOD_CONFIGS = {
    "base": {"temperature": 0.2, "num_comps": 1, "use_cot": False},
    "cot": {"temperature": 0.2, "num_comps": 1, "use_cot": True},
    "sc": {"temperature": 1.0, "num_comps": 5, "use_cot": False},
    "cot_sc": {"temperature": 1.0, "num_comps": 5, "use_cot": True},
    "debate": {"temperature": 0.7, "num_agents": 3, "num_rounds": 2, "use_cot": True},
}


def get_method_config(method: str) -> Dict[str, Any]:
    """
    Get configuration for a baseline method.

    Args:
        method: Method name (base, cot, sc, cot_sc, debate)

    Returns:
        Configuration dictionary
    """
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method: {method}. Choose from {list(METHOD_CONFIGS.keys())}")
    return METHOD_CONFIGS[method]


def format_debate_history(debate_history: List[Dict[str, Dict[str, Any]]], current_round: int) -> str:
    """
    Format debate history for prompt.

    Args:
        debate_history: List of round dictionaries, each containing agent responses
        current_round: Current round number (0-indexed)

    Returns:
        Formatted string of previous responses
    """
    if current_round == 0:
        return ""

    formatted = []
    for round_idx in range(current_round):
        round_data = debate_history[round_idx]
        formatted.append(f"--- Round {round_idx + 1} ---")
        for agent_id, agent_data in sorted(round_data.items()):
            response = agent_data.get('response', '')
            formatted.append(f"[{agent_id}]: {response}\n")

    return "\n".join(formatted)
