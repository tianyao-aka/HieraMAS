"""
Baseline experiments for evaluating LLM performance.

Methods supported:
- base: Direct LLM answering
- cot: Chain-of-thought prompting
- sc: Self-consistency (multiple samples + majority vote)
- cot_sc: CoT + self-consistency
"""

from .baseline_utils import (
    majority_vote,
    extract_mmlu_answer,
    extract_gsm8k_answer,
    extract_math_answer,
    extract_code,
    save_results,
)
