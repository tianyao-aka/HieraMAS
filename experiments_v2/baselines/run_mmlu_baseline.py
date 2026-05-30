"""
MMLU Baseline Experiments

Run baseline methods on the MMLU dataset:
- base: Direct LLM answering
- cot: Chain-of-thought prompting
- sc: Self-consistency (5 samples, majority vote)
- cot_sc: CoT + self-consistency
- debate: Multi-agent debate (3 agents, 2 rounds by default)

Usage:
    python experiments_v2/baselines/run_mmlu_baseline.py --method cot_sc --model qwen3-80b --limit 100
    python experiments_v2/baselines/run_mmlu_baseline.py --method debate --model gpt-4o-mini --num_agents 3 --num_rounds 2
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from HieraMAS.llm.llm_registry import LLMRegistry
from HieraMAS.utils.globals import Cost, PromptTokens, CompletionTokens
from experiments_v2.baselines.baseline_utils import (
    majority_vote,
    extract_answer_with_llm,
    verify_answer_with_llm,
    save_results,
    compute_accuracy,
    get_method_config,
    format_debate_history,
)


# MMLU Prompt Templates
MMLU_PROMPT_BASE = """Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}

Answer with only the letter (A, B, C, or D)."""

MMLU_PROMPT_COT = """Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}

Let's think step by step, then give your final answer as a single letter (A, B, C, or D)."""

MMLU_PROMPT_DEBATE_INITIAL = """Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}

You are participating in a multi-agent debate to find the correct answer. Think step by step and provide your reasoning, then give your answer as a single letter (A, B, C, or D)."""

MMLU_PROMPT_DEBATE_ROUND = """Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}

Previous responses from other agents:
{debate_history}

Consider the reasoning from other agents above. Think step by step, critically evaluate the arguments, and provide your refined answer as a single letter (A, B, C, or D)."""


def load_mmlu_dataset(data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load MMLU dataset from JSONL file.

    Args:
        data_path: Path to MMLU JSONL file
        limit: Maximum number of samples to load

    Returns:
        List of dataset records
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    print(f"Loading MMLU dataset from {data_path}")

    records = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Convert JSONL format to expected format
                choices = record.get('choices', [])
                records.append({
                    'question': record.get('question', ''),
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'C': choices[2] if len(choices) > 2 else '',
                    'D': choices[3] if len(choices) > 3 else '',
                    'correct_answer': record.get('correct_answer_letter', ''),
                    'subject': record.get('subject', ''),
                })

    print(f"Loaded {len(records)} samples")

    if limit:
        records = records[:limit]
        print(f"Limited to {len(records)} samples")

    return records


def build_prompt(record: Dict[str, Any], use_cot: bool = False) -> List[Dict[str, str]]:
    """
    Build prompt messages for MMLU question.

    Args:
        record: Dataset record with question and options
        use_cot: Whether to use chain-of-thought prompting

    Returns:
        List of message dictionaries
    """
    template = MMLU_PROMPT_COT if use_cot else MMLU_PROMPT_BASE
    user_content = template.format(
        question=record['question'],
        A=record['A'],
        B=record['B'],
        C=record['C'],
        D=record['D'],
    )

    return [
        {'role': 'system', 'content': 'You are a helpful assistant that answers multiple choice questions.'},
        {'role': 'user', 'content': user_content},
    ]


@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
async def run_single_query(
    llm,
    record: Dict[str, Any],
    use_cot: bool,
    temperature: float,
    num_comps: int,
) -> Dict[str, Any]:
    """
    Run a single query and return result.

    Args:
        llm: LLM instance
        record: Dataset record
        use_cot: Whether to use CoT prompting
        temperature: Sampling temperature
        num_comps: Number of completions

    Returns:
        Result dictionary
    """
    messages = build_prompt(record, use_cot=use_cot)

    # Get response(s) - call LLM multiple times for self-consistency (in parallel)
    if num_comps > 1:
        tasks = [
            llm.agen(messages=messages, temperature=temperature, num_comps=1)
            for _ in range(num_comps)
        ]
        responses = await asyncio.gather(*tasks)
        responses = list(responses)  # Convert tuple to list
    else:
        resp = await llm.agen(messages=messages, temperature=temperature, num_comps=1)
        responses = [resp]

    # Extract answers from responses using LLM
    judge_llm = LLMRegistry.get("gpt-4o-mini")
    extract_tasks = [extract_answer_with_llm(r, dataset="mmlu", llm=judge_llm) for r in responses]
    extracted_answers = await asyncio.gather(*extract_tasks)
    extracted_answers = list(extracted_answers)

    # Get final answer (majority vote if multiple)
    if len(extracted_answers) > 1:
        final_answer = majority_vote(extracted_answers)
    else:
        final_answer = extracted_answers[0] if extracted_answers else ""

    # Check correctness using LLM
    correct_answer = record['correct_answer']
    is_correct = await verify_answer_with_llm(final_answer, correct_answer, llm=judge_llm)

    return {
        'question': record['question'],
        'options': {
            'A': record['A'],
            'B': record['B'],
            'C': record['C'],
            'D': record['D'],
        },
        'correct_answer': correct_answer,
        'predicted_answer': final_answer,
        'is_correct': is_correct,
        'responses': responses,
        'extracted_answers': extracted_answers,
    }


@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
async def run_debate_query(
    llm,
    record: Dict[str, Any],
    num_agents: int = 3,
    num_rounds: int = 2,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run multi-agent debate on a single query.

    Args:
        llm: LLM instance
        record: Dataset record
        num_agents: Number of agents participating in debate
        num_rounds: Number of debate rounds (total rounds including initial)
        temperature: Sampling temperature

    Returns:
        Result dictionary with debate history
    """
    debate_history = []
    
    # Run debate rounds
    for round_idx in range(num_rounds):
        round_responses = {}
        
        for agent_idx in range(num_agents):
            # Build prompt based on round
            if round_idx == 0:
                # Initial round: no history
                prompt_template = MMLU_PROMPT_DEBATE_INITIAL
                prompt = prompt_template.format(
                    question=record['question'],
                    A=record['A'],
                    B=record['B'],
                    C=record['C'],
                    D=record['D'],
                )
            else:
                # Subsequent rounds: include debate history
                history_text = format_debate_history(debate_history, round_idx)
                prompt_template = MMLU_PROMPT_DEBATE_ROUND
                prompt = prompt_template.format(
                    question=record['question'],
                    A=record['A'],
                    B=record['B'],
                    C=record['C'],
                    D=record['D'],
                    debate_history=history_text,
                )
            
            # Get agent response
            messages = [{"role": "user", "content": prompt}]
            response = await llm.agen(
                messages=messages,
                temperature=temperature,
                num_comps=1,
            )

            # Store response (extract answers later in batch)
            agent_id = f"agent_{agent_idx}"
            round_responses[agent_id] = {
                'response': response,
            }

        debate_history.append(round_responses)

    # Get final responses from last round and extract answers using LLM
    final_round = debate_history[-1]
    final_responses = [data['response'] for data in final_round.values()]

    judge_llm = LLMRegistry.get("gpt-4o-mini")
    extract_tasks = [extract_answer_with_llm(r, dataset="mmlu", llm=judge_llm) for r in final_responses]
    final_answers = await asyncio.gather(*extract_tasks)
    final_answers = list(final_answers)

    # Majority vote
    final_answer = majority_vote(final_answers)

    # Check correctness using LLM
    correct_answer = record['correct_answer']
    is_correct = await verify_answer_with_llm(final_answer, correct_answer, llm=judge_llm)

    return {
        'question': record['question'],
        'options': {
            'A': record['A'],
            'B': record['B'],
            'C': record['C'],
            'D': record['D'],
        },
        'correct_answer': correct_answer,
        'predicted_answer': final_answer,
        'is_correct': is_correct,
        'debate_history': debate_history,
        'final_answers': final_answers,
    }


async def run_baseline(
    method: str,
    model: str,
    dataset: List[Dict[str, Any]],
    output_dir: str = "result/baselines/",
    num_agents: int = 3,
    num_rounds: int = 2,
) -> Dict[str, Any]:
    """
    Run baseline experiment on MMLU dataset.

    Args:
        method: Baseline method (base, cot, sc, cot_sc, debate)
        model: Model name
        dataset: List of dataset records
        output_dir: Directory to save results
        num_agents: Number of agents for debate method
        num_rounds: Number of rounds for debate method

    Returns:
        Summary statistics
    """
    # Get method configuration
    config = get_method_config(method)
    temperature = config['temperature']

    # Handle method-specific configuration
    if method == 'debate':
        num_agents = config.get('num_agents', num_agents)
        num_rounds = config.get('num_rounds', num_rounds)
        print(f"\nRunning MMLU baseline:")
        print(f"  Method: {method}")
        print(f"  Model: {model}")
        print(f"  Temperature: {temperature}")
        print(f"  Num agents: {num_agents}")
        print(f"  Num rounds: {num_rounds}")
        print(f"  Dataset size: {len(dataset)}")
    else:
        num_comps = config['num_comps']
        use_cot = config['use_cot']
        print(f"\nRunning MMLU baseline:")
        print(f"  Method: {method}")
        print(f"  Model: {model}")
        print(f"  Temperature: {temperature}")
        print(f"  Num completions: {num_comps}")
        print(f"  Use CoT: {use_cot}")
        print(f"  Dataset size: {len(dataset)}")

    # Initialize LLM through OpenRouter only.
    llm = LLMRegistry.get(model)
    print(f"  Using OpenRouter backend for {model}")

    # Reset cost tracking
    Cost.instance().value = 0
    PromptTokens.instance().value = 0
    CompletionTokens.instance().value = 0

    results = []
    correct_count = 0

    # Process samples
    for i, record in enumerate(tqdm(dataset, desc="Processing")):
        try:
            if method == 'debate':
                result = await run_debate_query(
                    llm=llm,
                    record=record,
                    num_agents=num_agents,
                    num_rounds=num_rounds,
                    temperature=temperature,
                )
            else:
                result = await run_single_query(
                    llm=llm,
                    record=record,
                    use_cot=use_cot,
                    temperature=temperature,
                    num_comps=num_comps,
                )
            results.append(result)

            if result['is_correct']:
                correct_count += 1

            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                current_acc = correct_count / (i + 1)
                print(f"\n  Progress: {i+1}/{len(dataset)}, Accuracy: {current_acc:.2%}")

        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            results.append({
                'question': record['question'],
                'correct_answer': record['correct_answer'],
                'predicted_answer': '',
                'is_correct': False,
                'error': str(e),
            })

    # Compute summary
    summary = compute_accuracy(results)
    summary['total_cost'] = Cost.instance().value
    summary['prompt_tokens'] = PromptTokens.instance().value
    summary['completion_tokens'] = CompletionTokens.instance().value
    summary['method'] = method
    summary['model'] = model
    summary['temperature'] = temperature

    if method == 'debate':
        summary['num_agents'] = num_agents
        summary['num_rounds'] = num_rounds
    else:
        summary['num_comps'] = num_comps

    print(f"\n{'='*50}")
    print(f"Results Summary:")
    print(f"  Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total']})")
    print(f"  Total Cost: ${summary['total_cost']:.4f}")
    print(f"  Prompt Tokens: {summary['prompt_tokens']}")
    print(f"  Completion Tokens: {summary['completion_tokens']}")

    # Save results
    save_results(
        results=results,
        output_dir=output_dir,
        dataset_name="mmlu",
        model_name=model,
        method=method,
        summary=summary,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run MMLU baseline experiments")
    parser.add_argument(
        "--method",
        type=str,
        default="base",
        choices=["base", "cot", "sc", "cot_sc", "debate"],
        help="Baseline method to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        choices=["gpt-5-mini", "qwen3-80b"],
        help="Model to use",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Datasets/MMLU/test.jsonl",
        help="Path to MMLU dataset JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result/baselines/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=3,
        help="Number of agents for debate method",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=2,
        help="Number of debate rounds (including initial round)",
    )

    args = parser.parse_args()

    # Load dataset
    dataset = load_mmlu_dataset(
        data_path=args.dataset_path,
        limit=args.limit,
    )

    # Run baseline
    asyncio.run(run_baseline(
        method=args.method,
        model=args.model,
        dataset=dataset,
        output_dir=args.output_dir,
        num_agents=args.num_agents,
        num_rounds=args.num_rounds,
    ))


if __name__ == "__main__":
    main()
