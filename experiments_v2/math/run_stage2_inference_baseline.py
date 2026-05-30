"""
Baseline Inference Script for MATH Dataset.

This script provides a simplified baseline for comparison:
- NO workers (aggregator_only=True) - aggregator answers directly
- Fixed LLM for all agents (same as decision_llm)
- Simplified graph topology (random or fully connected)
- No voting (single pass, num_votes=1)
- No checkpoint loading required

This baseline measures the value of:
- Worker agents (vs aggregator-only)
- LLM selection learning
- Graph topology learning

Usage:
    # With random graph
    python experiments_v2/math/run_stage2_inference_baseline.py \
        --test_file Datasets/MATH/math_test.jsonl \
        --output_file result_v2/math/baseline/test_random.json \
        --random_graph \
        --max_samples 5

    # With full graph
    python experiments_v2/math/run_stage2_inference_baseline.py \
        --test_file Datasets/MATH/math_test.jsonl \
        --output_file result_v2/math/baseline/test_full.json \
        --full_graph \
        --max_samples 5
"""

import os
os.environ['TRUST_REMOTE_CODE'] = 'true'
os.environ['HF_HUB_TRUST_REMOTE_CODE'] = '1'

import sys
import copy
import asyncio
import argparse
import json
import pickle
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

from HieraMAS.graph.learnable_multi_node_graph import LearnableMultiNodeGraph, get_answer_checker
from HieraMAS.graph.random_graph_utils import generate_random_graph
from HieraMAS.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from HieraMAS.tools.reader.readers import JSONLReader
from Datasets.MATH import math_data_process


# =============================================================================
# Constants - Baseline Configuration
# =============================================================================

# Note: num_workers is set but will be bypassed by aggregator_only=True
DEFAULT_MULTI_NODE_CONFIGS = [
    {'role': 'Mathematical Analyst', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Math Solver', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Inspector', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
]


# =============================================================================
# Data Processing
# =============================================================================

def load_dataset(test_file: str, max_samples: int = None) -> List[Dict]:
    """Load and process MATH test dataset."""
    print(f"Loading dataset: {test_file}")

    if not Path(test_file).exists():
        raise FileNotFoundError(f"Dataset file not found: {test_file}")

    raw_data = JSONLReader.parse_file(test_file)
    processed = math_data_process(raw_data)

    if max_samples:
        processed = processed[:max_samples]

    print(f"Loaded {len(processed)} samples")
    return processed


# =============================================================================
# Graph Generation (Baseline)
# =============================================================================

def generate_full_graph(num_nodes: int) -> torch.Tensor:
    """Generate fully connected graph (all edges = 1, no self-loops)."""
    adj = torch.ones(num_nodes, num_nodes)
    # Remove self-loops
    adj.fill_diagonal_(0)
    return adj.flatten()


def generate_candidate_graphs(
    num_graphs: int,
    num_nodes: int,
    min_density: float = 0.3,
    max_density: float = 0.7,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Generate random graph candidates."""
    rng = random.Random(seed) if seed is not None else random.Random()

    graphs = []
    seen_hashes = set()
    attempts = 0
    max_attempts = num_graphs * 10

    while len(graphs) < num_graphs and attempts < max_attempts:
        attempts += 1
        graph_seed = rng.randint(0, 2**31 - 1)
        graph = generate_random_graph(num_nodes, min_density, max_density, seed=graph_seed)

        # Ensure diversity
        graph_hash = tuple(graph.tolist())
        if graph_hash in seen_hashes:
            continue
        seen_hashes.add(graph_hash)

        graphs.append(graph)

    return graphs


# =============================================================================
# Baseline Inference
# =============================================================================

async def baseline_inference(
    graph: LearnableMultiNodeGraph,
    task: str,
    answer: str,
    graph_adj: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run single-pass baseline inference.

    No voting, no role skipping, just direct execution.
    """
    input_dict = {'task': task}

    # Execute graph
    result = await graph.forward(
        input_dict,
        ground_truth=answer,
        hard=True,
        is_training=False,
        force_skip_roles=set(),  # No role skipping in baseline
        override_spatial_adj=graph_adj.to(device),
    )

    # Check answer using the domain's answer checker
    final_prediction = result['final_prediction']
    answer_checker = get_answer_checker('math')
    is_correct = await answer_checker(final_prediction, answer)

    # Extract costs
    total_cost = result.get('total_cost', 0)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for role, node_result in result.get('node_results', {}).items():
        total_prompt_tokens += node_result.get('prompt_tokens', 0)
        total_completion_tokens += node_result.get('completion_tokens', 0)

    return {
        'final_prediction': final_prediction,
        'is_correct': is_correct,
        'total_cost': total_cost,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
    }


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Baseline Inference for MATH (no workers, fixed LLM, simple graph)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # No checkpoint arguments needed for baseline

    # Dataset
    parser.add_argument('--test_file', type=str,
                        default='Datasets/MATH/math_test.jsonl',
                        help='Path to test dataset')
    parser.add_argument('--output_file', type=str,
                        default='result_v2/math/baseline/test_results.json',
                        help='Path to save results')

    # Graph mode (required: one of random_graph or full_graph)
    parser.add_argument('--random_graph', action='store_true',
                        help='Use random graph topology')
    parser.add_argument('--full_graph', action='store_true',
                        help='Use fully connected graph topology')

    # Graph parameters (for random graph)
    parser.add_argument('--graph_min_density', type=float, default=0.3,
                        help='Minimum edge density for random graphs')
    parser.add_argument('--graph_max_density', type=float, default=0.7,
                        help='Maximum edge density for random graphs')

    # LLM configuration
    parser.add_argument('--decision_llm', type=str, default='gpt-5-mini',
                        help='Fixed LLM for all agents (baseline uses same LLM everywhere)')

    # Processing
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of test samples to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Graph configuration
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='Decision node method')
    parser.add_argument('--mode', type=str, default='FullConnected',
                        help='Topology mode (for initialization, overridden by graph)')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='Number of rounds')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')

    # API key
    parser.add_argument('--openrouter_key_variant', type=int, default=1,
                        choices=[1, 2],
                        help='OpenRouter API key variant')

    # Debug
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')

    # Parallel processing
    parser.add_argument('--parallel_batch_size', type=int, default=10,
                        help='Number of samples to process in parallel')

    # Checkpoint and resume
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint file (pickle)')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Save checkpoint every N batches')

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_summary(all_results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from all results."""
    total_samples = len(all_results)
    solved_count = sum(1 for r in all_results if r['solved'])
    total_cost = sum(r['cost'] for r in all_results)
    total_prompt_tokens = sum(r['prompt_tokens'] for r in all_results)
    total_completion_tokens = sum(r['completion_tokens'] for r in all_results)

    return {
        'solve_rate': solved_count / total_samples if total_samples > 0 else 0.0,
        'total_samples': total_samples,
        'solved_count': solved_count,
        'total_cost': total_cost,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'mean_cost_per_sample': total_cost / total_samples if total_samples > 0 else 0.0,
        'mean_prompt_tokens_per_sample': total_prompt_tokens / total_samples if total_samples > 0 else 0.0,
        'mean_completion_tokens_per_sample': total_completion_tokens / total_samples if total_samples > 0 else 0.0,
    }


async def process_single_sample(
    idx: int,
    sample: Dict[str, Any],
    graph: LearnableMultiNodeGraph,
    graph_adj: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Process a single sample with baseline inference."""
    task = sample['task']
    answer = sample['answer']

    # Run baseline inference
    result = await baseline_inference(
        graph, task, answer, graph_adj, device
    )

    return {
        'idx': idx,
        'task': task,
        'ground_truth': answer,
        'level': sample.get('level', 'unknown'),
        'type': sample.get('type', 'unknown'),
        'final_prediction': result['final_prediction'],
        'solved': result['is_correct'],
        'cost': result['total_cost'],
        'prompt_tokens': result['total_prompt_tokens'],
        'completion_tokens': result['total_completion_tokens'],
        'graph_type': 'full' if args.full_graph else 'random',
    }


async def main():
    """Main baseline inference entry point."""
    args = parse_args()

    # Validate graph mode
    if args.random_graph and args.full_graph:
        raise ValueError("Cannot use both --random_graph and --full_graph")
    if not args.random_graph and not args.full_graph:
        raise ValueError("Baseline requires --random_graph or --full_graph")

    # Set OpenRouter key variant
    from HieraMAS.utils.globals import OpenRouterKeyVariant
    OpenRouterKeyVariant.instance().value = args.openrouter_key_variant

    # Set random seeds
    set_random_seeds(args.seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Time.instance().value = timestamp

    graph_type = "full" if args.full_graph else "random"

    print("=" * 80)
    print("BASELINE INFERENCE (MATH)")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Test file: {args.test_file}")
    print(f"Graph type: {graph_type}")
    print(f"Decision LLM: {args.decision_llm} (used for ALL agents)")
    print(f"Mode: aggregator_only=True (no workers)")
    print(f"Voting: num_votes=1 (single pass)")
    print(f"Parallel batch size: {args.parallel_batch_size}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test dataset
    test_dataset = load_dataset(args.test_file, args.max_samples)

    # Get number of nodes from configs
    multi_node_configs = DEFAULT_MULTI_NODE_CONFIGS
    num_nodes = len(multi_node_configs)
    print(f"Number of nodes: {num_nodes}")

    # Generate graph topology
    print("\n[1] Generating graph topology...")
    if args.full_graph:
        graph_adj = generate_full_graph(num_nodes)
        print(f"  Generated fully connected graph (density=1.0)")
    else:
        graphs = generate_candidate_graphs(
            num_graphs=1,
            num_nodes=num_nodes,
            min_density=args.graph_min_density,
            max_density=args.graph_max_density,
            seed=args.seed,
        )
        graph_adj = graphs[0]
        density = graph_adj.sum().item() / (num_nodes * num_nodes)
        print(f"  Generated random graph (density={density:.2f})")

    # Initialize graph with baseline settings
    print("\n[2] Initializing LearnableMultiNodeGraph (baseline mode)...")
    graph = LearnableMultiNodeGraph(
        multi_node_configs=multi_node_configs,
        decision_method=args.decision_method,
        domain='gsm8k',  # MATH uses gsm8k domain (same prompt set)
        mode=args.mode,
        num_rounds=args.num_rounds,
        hidden_dim=args.hidden_dim,
        use_llm_for_task_meta=True,
        optimized_spatial=False,
        optimized_temporal=False,
        decision_llm=args.decision_llm,
        share_selector=True,
        enable_skip_learning=False,
        fixed_llm=True,  # Use same LLM for all agents
        aggregator_only=True,  # Skip workers, aggregator answers directly
    )
    graph.to(device)
    graph.eval()

    print(f"  fixed_llm=True (all agents use {args.decision_llm})")
    print(f"  aggregator_only=True (no worker agents)")

    # Skip checkpoint loading (baseline mode)
    print("\n[3] Skipping checkpoint loading (baseline mode)")

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Process samples in parallel batches
    print(f"\n[4] Processing {len(test_dataset)} samples (parallel_batch_size={args.parallel_batch_size})...")
    all_results = []
    batch_size = args.parallel_batch_size
    num_batches = (len(test_dataset) + batch_size - 1) // batch_size

    # Resume from checkpoint if specified
    start_batch = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        with open(args.resume_from, 'rb') as f:
            ckpt = pickle.load(f)
        start_batch = ckpt['batch_idx'] + 1
        all_results = ckpt['all_results']
        print(f"Resumed: batch_idx={ckpt['batch_idx']}, processed={len(all_results)} samples")

    for batch_idx in tqdm(range(start_batch, num_batches), desc="Inference batches",
                          initial=start_batch, total=num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(test_dataset))
        batch_samples = [(i, test_dataset[i]) for i in range(batch_start, batch_end)]

        # Process batch in parallel using asyncio.gather
        batch_results = await asyncio.gather(*[
            process_single_sample(idx, sample, copy.deepcopy(graph), graph_adj, args, device)
            for idx, sample in batch_samples
        ], return_exceptions=True)

        # Handle exceptions and collect results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                sample_idx = batch_start + i
                print(f"\nError processing sample {sample_idx}: {result}")
                all_results.append({
                    'idx': sample_idx,
                    'task': test_dataset[sample_idx]['task'],
                    'ground_truth': test_dataset[sample_idx]['answer'],
                    'level': test_dataset[sample_idx].get('level', 'unknown'),
                    'type': test_dataset[sample_idx].get('type', 'unknown'),
                    'final_prediction': 'ERROR',
                    'solved': False,
                    'cost': 0.0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'error': str(result),
                })
            else:
                all_results.append(result)

        # Progress update
        if args.verbose:
            solved_count = sum(1 for r in all_results if r.get('solved', False))
            current_cost = sum(r.get('cost', 0) for r in all_results)
            print(f"\nBatch {batch_idx + 1}/{num_batches}: "
                  f"processed={len(all_results)}, "
                  f"solve_rate={solved_count/len(all_results):.4f}, "
                  f"cost=${current_cost:.4f}")

        # Save checkpoint every interval
        if (batch_idx + 1) % args.checkpoint_interval == 0:
            ckpt_file = checkpoint_dir / f"inference_ckpt_batch_{batch_idx}.pkl"
            with open(ckpt_file, 'wb') as f:
                pickle.dump({
                    'batch_idx': batch_idx,
                    'all_results': all_results,
                    'args': vars(args),
                }, f)
            print(f"\nCheckpoint saved: {ckpt_file} ({len(all_results)} samples processed)")

    # Compute summary
    summary = compute_summary(all_results)

    # Build final results dict
    results_data = {
        'summary': summary,
        'config': {
            'baseline_mode': True,
            'aggregator_only': True,
            'fixed_llm': True,
            'num_votes': 1,
            'graph_type': graph_type,
            'test_file': args.test_file,
            'decision_llm': args.decision_llm,
            'parallel_batch_size': args.parallel_batch_size,
            'seed': args.seed,
            'timestamp': timestamp,
        },
        'instances': all_results,
    }

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Solve Rate: {summary['solve_rate']:.4f} ({summary['solved_count']}/{summary['total_samples']})")
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    print(f"Mean Cost/Sample: ${summary['mean_cost_per_sample']:.6f}")
    print(f"Total Prompt Tokens: {summary['total_prompt_tokens']:,}")
    print(f"Total Completion Tokens: {summary['total_completion_tokens']:,}")
    print(f"\nResults saved to: {args.output_file}")
    print("=" * 80)

    return results_data


if __name__ == "__main__":
    asyncio.run(main())
