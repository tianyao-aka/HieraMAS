"""
Stage 2 Test Inference Script for MATH Dataset.

This script performs test-time inference using both:
- Stage 1 checkpoint (LLM Selector)
- Stage 2 checkpoint (Graph Topology Classifier)

For each test sample:
1. Generate N random graph topologies
2. Score all graphs with Stage 2 GNN classifier
3. Select top-K graphs by score
4. Compute role skip probabilities (from Stage 1)
5. Run M inference passes with majority voting
6. Record detailed results

Usage:
    python experiments_v2/math/run_stage2_inference.py \
        --stage1_checkpoint result_v2/math/stage1/checkpoint/bestModel/best_valid_model.pt \
        --stage2_checkpoint result_v2/math/stage2/classifier/stage2_classifier_best.pt \
        --test_file Datasets/MATH/math_test.jsonl \
        --output_file result_v2/math/stage2/inference/test_results.json
"""

import os
os.environ['TRUST_REMOTE_CODE'] = 'true'
os.environ['HF_HUB_TRUST_REMOTE_CODE'] = '1'

import sys
import asyncio
import argparse
import copy
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter
from tqdm import tqdm
import pickle

import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

from HieraMAS.graph.learnable_multi_node_graph import LearnableMultiNodeGraph, get_answer_checker
from HieraMAS.graph.random_graph_utils import generate_random_graph
from HieraMAS.model.graph_classifier import GraphClassifier
from HieraMAS.model.embeddings import PositionLLMEmbedder
from HieraMAS.prompt.llm_profiles import get_llm_index, get_llm_name, LLM_OPTIONS
from HieraMAS.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from HieraMAS.tools.reader.readers import JSONLReader
from Datasets.MATH import math_data_process
from HieraMAS.llm.gpt_chat import achat

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MULTI_NODE_CONFIGS = [
    {'role': 'Mathematical Analyst', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Math Solver', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Inspector', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
]

ROLE_DESCRIPTIONS = [
    "Mathematical Analyst: Analyzes mathematical problems, identifies key concepts, and breaks down complex problems into solvable components.",
    "Math Solver: Executes step-by-step mathematical computations and derives solutions using algebraic and numerical methods.",
    "Inspector: Verifies solutions, checks calculations for errors, and validates mathematical reasoning.",
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
# Checkpoint Loading
# =============================================================================

def load_graph_state(graph: LearnableMultiNodeGraph, graph_state: Dict):
    """Load graph state from checkpoint (same as run_stage2_datagen.py)."""
    # Load spatial/temporal logits
    if 'spatial_logits' in graph_state:
        graph.spatial_logits.data = graph_state['spatial_logits'].to(graph.spatial_logits.device)
    if 'temporal_logits' in graph_state:
        graph.temporal_logits.data = graph_state['temporal_logits'].to(graph.temporal_logits.device)

    # Load shared selector state (handles both key names for compatibility)
    if 'shared_selector_state' in graph_state and graph.shared_selector is not None:
        graph.shared_selector.load_state_dict(graph_state['shared_selector_state'])
    elif 'shared_selector' in graph_state and graph.shared_selector is not None:
        graph.shared_selector.load_state_dict(graph_state['shared_selector'])

    # Load embedder state
    if 'embedder_state' in graph_state:
        embedder_state = graph_state['embedder_state']
        if hasattr(graph, 'embedder'):
            if '_llm_embeddings_cache' in embedder_state:
                graph.embedder._llm_embeddings_cache = embedder_state['_llm_embeddings_cache']

    # Load GCN and edge scorer state if they exist
    if 'simple_gcn_state' in graph_state and hasattr(graph, 'simple_gcn') and graph.simple_gcn is not None:
        graph.simple_gcn.load_state_dict(graph_state['simple_gcn_state'])
    if 'edge_scorer_state' in graph_state and hasattr(graph, 'edge_scorer') and graph.edge_scorer is not None:
        graph.edge_scorer.load_state_dict(graph_state['edge_scorer_state'])

    # Handle masks
    if 'spatial_masks' in graph_state:
        graph.spatial_masks.data = graph_state['spatial_masks']
    if 'temporal_masks' in graph_state:
        graph.temporal_masks.data = graph_state['temporal_masks']


def load_stage1_checkpoint(
    checkpoint_path: str,
    graph: LearnableMultiNodeGraph,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load Stage 1 LLM Selector weights into graph.

    Args:
        checkpoint_path: Path to Stage 1 checkpoint file
        graph: LearnableMultiNodeGraph to load weights into
        device: Target device

    Returns:
        Config dict from checkpoint
    """
    print(f"Loading Stage 1 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    graph_state = checkpoint.get('graph_state', {})
    config = checkpoint.get('config', {})

    load_graph_state(graph, graph_state)

    print(f"  Checkpoint iteration: {checkpoint.get('iteration', 'N/A')}")
    print(f"  Checkpoint solve rate: {checkpoint.get('solve_rate', checkpoint.get('valid_solve_rate', 'N/A'))}")

    return config


def load_stage2_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_nodes: int = 3,
) -> Tuple[GraphClassifier, Dict[str, Any]]:
    """
    Load Stage 2 Graph Classifier.

    Args:
        checkpoint_path: Path to Stage 2 checkpoint file
        device: Target device
        num_nodes: Number of nodes in the graph

    Returns:
        Tuple of (classifier, metadata)
    """
    print(f"Loading Stage 2 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    metadata = checkpoint.get('metadata', {})
    args = checkpoint.get('args', {})

    # Extract architecture parameters from checkpoint
    gcn_hidden_dim = args.get('gcn_hidden_dim', 256)
    gcn_output_dim = args.get('gcn_output_dim', 128)
    classifier_hidden_dim = args.get('classifier_hidden_dim', 64)
    dropout = args.get('dropout', 0.05)

    # Use num_nodes from metadata if available
    num_nodes = metadata.get('num_nodes', num_nodes)

    # Initialize classifier with same architecture
    classifier = GraphClassifier(
        num_nodes=num_nodes,
        gcn_hidden_dim=gcn_hidden_dim,
        gcn_output_dim=gcn_output_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout=dropout,
        role_descriptions=ROLE_DESCRIPTIONS,
    )

    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier.eval()

    print(f"  Loaded GraphClassifier (num_nodes={num_nodes})")

    return classifier, metadata


# =============================================================================
# Graph Generation and Selection
# =============================================================================

def generate_candidate_graphs(
    num_graphs: int,
    num_nodes: int,
    min_density: float = 0.3,
    max_density: float = 0.7,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Generate a pool of random graph candidates.

    Note: generate_graph_pool caps at 100, so we generate manually for larger pools.

    Args:
        num_graphs: Number of graphs to generate
        num_nodes: Number of nodes in each graph
        min_density: Minimum edge density
        max_density: Maximum edge density
        seed: Random seed for reproducibility

    Returns:
        List of flattened adjacency tensors, each of shape (N*N,)
    """
    # Use local Random instance to avoid global state conflicts in parallel execution
    rng = random.Random(seed) if seed is not None else random.Random()

    graphs = []
    seen_hashes = set()
    attempts = 0
    max_attempts = num_graphs * 10

    while len(graphs) < num_graphs and attempts < max_attempts:
        attempts += 1
        # Generate a unique seed for each graph attempt using local RNG
        graph_seed = rng.randint(0, 2**31 - 1)
        graph = generate_random_graph(num_nodes, min_density, max_density, seed=graph_seed)

        # Ensure diversity
        graph_hash = tuple(graph.tolist())
        if graph_hash in seen_hashes:
            continue
        seen_hashes.add(graph_hash)

        graphs.append(graph)

    if len(graphs) < num_graphs:
        print(f"Warning: Could only generate {len(graphs)} diverse graphs (requested {num_graphs})")

    return graphs


def select_top_k_graphs_batch(
    classifier: GraphClassifier,
    task: str,
    candidate_graphs: List[torch.Tensor],
    top_k: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Score candidate graphs with batch GNN forward and select top-k.

    This is more efficient than the sequential version as it:
    1. Computes task embedding once
    2. Uses batch forward to score all graphs together

    Args:
        classifier: Trained GraphClassifier
        task: Task text
        candidate_graphs: List of flattened adjacency tensors (N*N,)
        top_k: Number of top graphs to return
        device: Target device

    Returns:
        Tuple of (top_k_graphs, top_k_scores)
    """
    if not candidate_graphs:
        return [], []

    # Pre-compute task embedding once
    task_emb = classifier.get_task_embedding(task, device)

    # Stack all graphs for batch processing
    stacked_graphs = torch.stack(candidate_graphs).to(device)  # (N_graphs, N*N)

    # Batch forward
    classifier.eval()
    with torch.no_grad():
        logits = classifier.forward_batch_efficient(task, stacked_graphs, task_emb)
        scores = torch.sigmoid(logits).squeeze(-1)  # (N_graphs,)

    # Get top-k using torch.topk (more efficient than sorting)
    top_k_actual = min(top_k, len(candidate_graphs))
    top_scores, top_indices = torch.topk(scores, top_k_actual)

    # Convert to lists
    top_k_graphs = [candidate_graphs[i] for i in top_indices.tolist()]
    top_k_scores = top_scores.tolist()

    return top_k_graphs, top_k_scores


# =============================================================================
# Majority Voting Inference
# =============================================================================

def extract_llm_selections(node_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract LLM selections from node results.

    Args:
        node_results: Dict mapping role -> node result dict

    Returns:
        Dict mapping role -> {workers: [llm_names], aggregator: llm_name}
    """
    llm_selections = {}

    for role, result in node_results.items():
        if 'llm_selection' in result:
            llm_list = result['llm_selection']
            # Format: [worker1, worker2, ..., aggregator]
            if llm_list:
                llm_selections[role] = {
                    'workers': llm_list[:-1],
                    'aggregator': llm_list[-1] if llm_list else 'unknown',
                }

    return llm_selections



async def llm_majority_vote(all_predictions: List[str], ground_truth: str) -> Tuple[str, bool]:
    """Use gpt-4o-mini to determine majority answer from multiple predictions.

    Args:
        all_predictions: List of prediction strings (may contain reasoning + answer)
        ground_truth: The correct answer for verification

    Returns:
        Tuple of (final_prediction, is_correct)
    """

    predictions_text = "\n".join([f"[{i+1}] {pred}" for i, pred in enumerate(all_predictions)])

    prompt = f"""Given these {len(all_predictions)} predictions for a math problem, determine the majority answer. think step by step.

Predictions:
{predictions_text}

Ground Truth: {ground_truth}

Tasks:
1. Extract the final numerical answer from each prediction
2. Determine which answer appears most frequently (majority vote)
3. Check if the majority answer matches the ground truth

Reply in this exact format:
EXTRACTED_ANSWERS: <list each prediction's extracted answer, e.g., "1: 84, 2: 84, 3: 42">
MAJORITY_ANSWER: <the majority numerical answer>
MAJORITY_INDEX: <index of first prediction with majority answer, e.g., 1>
CORRECT: <yes or no>"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await achat("gpt-4o-mini", messages)

        # Parse correctness
        is_correct = "correct: yes" in response.lower()

        # Parse majority index to select the corresponding prediction
        final_prediction = all_predictions[0]  # default fallback
        for line in response.split('\n'):
            if 'MAJORITY_INDEX:' in line.upper():
                try:
                    idx = int(line.split(':')[1].strip()) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(all_predictions):
                        final_prediction = all_predictions[idx]
                except (ValueError, IndexError):
                    pass
                break

        return final_prediction, is_correct

    except Exception as e:
        # Fallback: return first prediction and use existing checker
        from HieraMAS.graph.learnable_multi_node_graph import get_answer_checker
        answer_checker = get_answer_checker('math')
        is_correct = await answer_checker(all_predictions[0], ground_truth)
        return all_predictions[0], is_correct


async def inference_with_majority_vote(
    graph: LearnableMultiNodeGraph,
    task: str,
    answer: str,
    top_k_graphs: List[torch.Tensor],
    roles_to_skip: Set[str],
    num_votes: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run inference multiple times and use majority voting.

    For each vote:
    1. Cycle through top-k graphs
    2. Execute graph with that topology
    3. Record prediction

    Final prediction is the most common answer.

    Args:
        graph: LearnableMultiNodeGraph with loaded weights
        task: Task text
        answer: Ground truth answer
        top_k_graphs: List of top-k graph topologies
        roles_to_skip: Set of roles to skip
        num_votes: Number of runs for voting
        device: Target device

    Returns:
        Dict with final_prediction, all_predictions, per_run_details, etc.
    """
    input_dict = {'task': task}
    all_predictions = []
    all_results = []
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for vote_idx in range(num_votes):
        # Cycle through top-k graphs
        graph_idx = vote_idx % len(top_k_graphs)
        graph_adj = top_k_graphs[graph_idx].to(device)

        # Execute with role skip and overridden graph
        result = await graph.forward(
            input_dict,
            ground_truth=answer,
            hard=True,
            is_training=False,
            force_skip_roles=roles_to_skip,
            override_spatial_adj=graph_adj,
        )

        all_predictions.append(result['final_prediction'])
        all_results.append(result)
        total_cost += result.get('total_cost', 0)

        # Accumulate tokens from node_results
        for role, node_result in result.get('node_results', {}).items():
            total_prompt_tokens += node_result.get('prompt_tokens', 0)
            total_completion_tokens += node_result.get('completion_tokens', 0)

    # Majority voting using LLM to extract and compare answers
    prediction_counts = Counter(all_predictions)
    final_prediction, final_solved = await llm_majority_vote(all_predictions, answer)

    # Extract LLM selections from first run (representative)
    llm_selections = {}
    if all_results:
        llm_selections = extract_llm_selections(all_results[0].get('node_results', {}))

    return {
        'final_prediction': final_prediction,
        'final_solved': final_solved,
        'all_predictions': all_predictions,
        'vote_counts': dict(prediction_counts),
        'total_cost': total_cost,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'top_1_graph': top_k_graphs[0].tolist(),
        'llm_selections': llm_selections,
    }


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Stage 2 Test Inference with Graph Classifier for MATH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required checkpoints
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Path to Stage 1 LLM Selector checkpoint')
    parser.add_argument('--stage2_checkpoint', type=str, required=True,
                        help='Path to Stage 2 Graph Classifier checkpoint')

    # Dataset
    parser.add_argument('--test_file', type=str,
                        default='Datasets/MATH/math_test.jsonl',
                        help='Path to test dataset')
    parser.add_argument('--output_file', type=str,
                        default='result_v2/math/stage2/inference/test_results.json',
                        help='Path to save results')

    # Graph selection parameters
    parser.add_argument('--num_random_graphs', type=int, default=200,
                        help='Number of random graphs to generate per sample')
    parser.add_argument('--top_k_graphs', type=int, default=5,
                        help='Number of top graphs to select for inference')
    parser.add_argument('--graph_min_density', type=float, default=0.3,
                        help='Minimum edge density for random graphs')
    parser.add_argument('--graph_max_density', type=float, default=0.7,
                        help='Maximum edge density for random graphs')

    # Inference parameters
    parser.add_argument('--num_votes', type=int, default=1,
                        help='Number of runs for majority voting')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for all agents during inference')
    parser.add_argument('--topK_roles_removed', type=int, default=0,
                        help='Maximum number of roles to skip (only those where skip is argmax)')
    parser.add_argument('--rotation_top_n', type=int, default=2,
                        help='Number of top LLMs to rotate for topN_rotation strategy')

    # LLM configuration
    parser.add_argument('--use_qwen3_80b', action='store_true',
                        help='Exclude GPT models from LLM selection')
    parser.add_argument('--exclude_llms', type=str, nargs='+',
                        default=['deepseek-r1-14b'],
                        help='LLM names to exclude from selection')

    # Processing
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of test samples to process')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')

    # Graph configuration (must match Stage 1)
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='Decision node method')
    parser.add_argument('--decision_llm', type=str, default='gpt-5-mini',
                        help='Fixed LLM for decision node')
    parser.add_argument('--mode', type=str, default='FullConnected',
                        help='Topology mode')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='Number of rounds')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--enable_skip_learning', action='store_true', default=True,
                        help='Enable skip learning')
    parser.add_argument('--keep_top_k', type=int, default=3,
                        help='Keep top-k roles')
    parser.add_argument('--share_selector', action='store_true', default=True,
                        help='Use shared selector')

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
                        help='Save checkpoint every N batches (default: 1 = every batch)')

    # Ablation flags
    parser.add_argument('--random_graph', action='store_true',
                        help='Use single random graph without GNN scoring (num_graphs=1, num_votes=1)')
    parser.add_argument('--fixed_llm', action='store_true',
                        help='Use decision_llm for all workers/aggregators (no LLM selection)')

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
    classifier: Optional[GraphClassifier],
    args: argparse.Namespace,
    device: torch.device,
    num_nodes: int,
) -> Dict[str, Any]:
    """
    Process a single sample - encapsulates the full inference pipeline.

    This function is designed to be called in parallel via asyncio.gather.

    Args:
        idx: Sample index
        sample: Sample dict with 'task' and 'answer'
        graph: LearnableMultiNodeGraph with loaded Stage 1 weights
        classifier: GraphClassifier with loaded Stage 2 weights (None if --random_graph)
        args: Parsed command-line arguments
        device: Target device
        num_nodes: Number of nodes in the graph

    Returns:
        Dict with inference results for this sample
    """
    task = sample['task']
    answer = sample['answer']
    input_dict = {'task': task}

    # Generate random graphs
    candidate_graphs = generate_candidate_graphs(
        num_graphs=args.num_random_graphs,
        num_nodes=num_nodes,
        min_density=args.graph_min_density,
        max_density=args.graph_max_density,
        seed=args.seed + idx,  # Different seed per sample for diversity
    )

    # Score and select top-k graphs (skip GNN scoring if --random_graph)
    if args.random_graph:
        top_k_graphs = candidate_graphs[:1]
        top_k_scores = [0.0]
    else:
        top_k_graphs, top_k_scores = select_top_k_graphs_batch(
            classifier, task, candidate_graphs, args.top_k_graphs, device
        )

    # Compute roles to skip (Stage 1 skip probabilities)
    role_skip_info = graph.compute_role_skip_probs(input_dict)
    roles_to_skip = graph.determine_roles_to_skip(
        role_skip_info,
        topK_roles_removed=args.topK_roles_removed
    )

    # Run inference with majority voting
    result = await inference_with_majority_vote(
        graph, task, answer,
        top_k_graphs, roles_to_skip,
        args.num_votes, device
    )

    # Return instance result
    return {
        'idx': idx,
        'task': task,
        'ground_truth': answer,
        'level': sample.get('level', 'unknown'),
        'type': sample.get('type', 'unknown'),
        'final_prediction': result['final_prediction'],
        'solved': result['final_solved'],
        'cost': result['total_cost'],
        'prompt_tokens': result['total_prompt_tokens'],
        'completion_tokens': result['total_completion_tokens'],
        'top_1_graph': result['top_1_graph'],
        'top_k_graph_scores': top_k_scores,
        'roles_skipped': list(roles_to_skip),
        'role_skip_info': {
            k: {'skip_prob': v['aggregator_skip_prob'], 'is_argmax': v['is_skip_argmax']}
            for k, v in role_skip_info.items()
        },
        'llm_selections': result['llm_selections'],
        'vote_counts': result['vote_counts'],
        'all_predictions': result['all_predictions'],
    }


async def main():
    """Main inference entry point."""
    args = parse_args()

    # Set OpenRouter key variant
    from HieraMAS.utils.globals import OpenRouterKeyVariant
    OpenRouterKeyVariant.instance().value = args.openrouter_key_variant

    # Set random seeds
    set_random_seeds(args.seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Time.instance().value = timestamp

    # Handle --random_graph flag
    if args.random_graph:
        args.num_random_graphs = 1
        args.top_k_graphs = 1
        args.num_votes = 1
        print("[random_graph] Using 1 random graph, 1 vote, no GNN scoring")

    print("=" * 80)
    print("STAGE 2 TEST INFERENCE (MATH)")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Stage 1 checkpoint: {args.stage1_checkpoint}")
    print(f"Stage 2 checkpoint: {args.stage2_checkpoint}")
    print(f"Test file: {args.test_file}")
    print(f"Num random graphs: {args.num_random_graphs}")
    print(f"Top-K graphs: {args.top_k_graphs}")
    print(f"Num votes: {args.num_votes}")
    print(f"Temperature: {args.temperature}")
    print(f"TopK roles removed: {args.topK_roles_removed}")
    print(f"Use Qwen3-80b only: {args.use_qwen3_80b}")
    print(f"Parallel batch size: {args.parallel_batch_size}")
    print(f"Seed: {args.seed}")
    print(f"Random graph (ablation): {args.random_graph}")
    print(f"Fixed LLM (ablation): {args.fixed_llm}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test dataset
    test_dataset = load_dataset(args.test_file, args.max_samples)

    # Get number of nodes from configs
    multi_node_configs = DEFAULT_MULTI_NODE_CONFIGS
    num_nodes = len(multi_node_configs)

    # Handle use_qwen3_80b flag
    if args.use_qwen3_80b:
        qwen3_80b_excludes = ['gpt-5-nano', 'gpt-5-mini', 'gpt-4o-mini', 'deepseek-r1-14b']
        if args.exclude_llms:
            args.exclude_llms = list(set(args.exclude_llms + qwen3_80b_excludes))
        else:
            args.exclude_llms = qwen3_80b_excludes
        # Also set decision LLM to qwen3-80b if not explicitly set
        if args.decision_llm == 'gpt-5-mini':  # default value
            args.decision_llm = 'qwen3-80b'
        print(f"[use_qwen3_80b] Excluding GPT models, using decision_llm={args.decision_llm}")

    # Convert excluded LLM names to indices
    excluded_llm_indices = None
    if args.exclude_llms:
        excluded_llm_indices = [get_llm_index(name) for name in args.exclude_llms]
        print(f"Excluding LLMs: {args.exclude_llms} (indices: {excluded_llm_indices})")

    # Initialize graph (Stage 1)
    print("\n[1] Initializing LearnableMultiNodeGraph...")
    graph = LearnableMultiNodeGraph(
        multi_node_configs=multi_node_configs,
        decision_method=args.decision_method,
        domain='gsm8k',
        mode=args.mode,
        num_rounds=args.num_rounds,
        hidden_dim=args.hidden_dim,
        use_llm_for_task_meta=True,
        optimized_spatial=False,  # We override with classifier-selected graphs
        optimized_temporal=False,
        decision_llm=args.decision_llm,
        share_selector=args.share_selector,
        inference_strategy='topN_rotation',
        rotation_top_n=args.rotation_top_n,
        use_qwen3_80b=args.use_qwen3_80b,
        enable_skip_learning=args.enable_skip_learning,
        keep_top_k=args.keep_top_k,
        excluded_llm_indices=excluded_llm_indices,
        fixed_llm=args.fixed_llm,
    )
    graph.to(device)

    # Load Stage 1 checkpoint
    print("\n[2] Loading Stage 1 checkpoint...")
    stage1_config = load_stage1_checkpoint(args.stage1_checkpoint, graph, device)

    # Set to evaluation mode
    graph.eval()

    # Set temperature for edge sampling
    graph.set_edge_temperature(args.temperature)
    print(f"  Set edge_temperature={args.temperature}")

    # Load Stage 2 classifier (skip if --random_graph)
    if not args.random_graph:
        print("\n[3] Loading Stage 2 classifier...")
        classifier, stage2_metadata = load_stage2_checkpoint(args.stage2_checkpoint, device, num_nodes)
    else:
        classifier = None
        print("\n[3] Skipping Stage 2 classifier (--random_graph)")

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
        # Use deepcopy to avoid state corruption from shared graph instance
        batch_results = await asyncio.gather(*[
            process_single_sample(idx, sample, copy.deepcopy(graph), classifier, args, device, num_nodes)
            for idx, sample in batch_samples
        ], return_exceptions=True)

        # Handle exceptions and collect results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                sample_idx = batch_start + i
                print(f"\nError processing sample {sample_idx}: {result}")
                # Create a placeholder result for failed samples
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
            'stage1_checkpoint': args.stage1_checkpoint,
            'stage2_checkpoint': args.stage2_checkpoint,
            'test_file': args.test_file,
            'num_random_graphs': args.num_random_graphs,
            'top_k_graphs': args.top_k_graphs,
            'num_votes': args.num_votes,
            'temperature': args.temperature,
            'use_qwen3_80b': args.use_qwen3_80b,
            'topK_roles_removed': args.topK_roles_removed,
            'exclude_llms': args.exclude_llms,
            'parallel_batch_size': args.parallel_batch_size,
            'seed': args.seed,
            'timestamp': timestamp,
            'random_graph': args.random_graph,
            'fixed_llm': args.fixed_llm,
        },
        'instances': all_results,
    }

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
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
