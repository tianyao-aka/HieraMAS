"""
Stage 2 Data Generation Script for MATH Dataset.

This script generates labeled data for training the graph classifier in Stage 2.
For each task, it:
1. Loads the Stage 1 trained LLM Selector
2. Generates K random graphs
3. Executes each graph on the task using the fixed LLM Selector
4. Collects rewards and labels top-2 as positive (if reward > 0)
5. Saves the dataset as a pickle file

Usage:
    python experiments_v2/math/run_stage2_datagen.py \
        --stage1_checkpoint result_v2/math/stage1/checkpoint/bestModel/best_valid_model.pt \
        --dataset_json Datasets/MATH/math_train.jsonl \
        --output_file result_v2/math/stage2/math_stage2_dataset.pkl \
        --num_graphs_per_task 5

Output format (pickle):
    List of dicts, each containing:
    - 'task': str - Task text
    - 'task_emb': torch.Tensor - Pre-computed task embedding (1024,)
    - 'graph': torch.Tensor - Flattened adjacency matrix (N*N,)
    - 'reward': float - Execution reward
    - 'label': int - 1 if positive (top-2 with reward > 0), 0 otherwise
    - 'solved': bool - Whether the task was solved
"""

import os
import copy
os.environ['TRUST_REMOTE_CODE'] = 'true'
os.environ['HF_HUB_TRUST_REMOTE_CODE'] = '1'

import sys
import asyncio
import argparse
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

from HieraMAS.graph.learnable_multi_node_graph import LearnableMultiNodeGraph
from HieraMAS.graph.random_graph_utils import generate_random_graph
from HieraMAS.model.embeddings import PositionLLMEmbedder
from HieraMAS.prompt.llm_profiles import get_llm_index
from HieraMAS.utils.const import HIERAMAS_ROOT
from HieraMAS.utils.globals import Time, Cost
from HieraMAS.tools.reader.readers import JSONLReader
from Datasets.MATH import math_data_process


# Default multi-node configurations for MATH (3 roles)
DEFAULT_MULTI_NODE_CONFIGS = [
    {'role': 'Mathematical Analyst', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Math Solver', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Inspector', 'num_workers': 2, 'agent_name': 'MathSolver', 'aggregator_method': 'FinalMultiAgentAggregator'},
]


def load_dataset(dataset_json: str, domain: str = 'math', max_samples: int = None) -> List[Dict]:
    """Load dataset from JSONL file."""
    print(f"Loading dataset: {dataset_json}")
    data = JSONLReader.parse_file(dataset_json)

    if domain == 'math':
        samples = math_data_process(data)
    else:
        # For other domains, assume data is already in correct format
        samples = data

    if max_samples:
        samples = samples[:max_samples]
    return samples


def load_graph_state(graph: LearnableMultiNodeGraph, graph_state: Dict):
    """Load graph state from checkpoint (same as run_math_learnable_graph.py)."""
    # Load spatial/temporal logits
    if 'spatial_logits' in graph_state:
        graph.spatial_logits.data = graph_state['spatial_logits'].to(graph.spatial_logits.device)
    if 'temporal_logits' in graph_state:
        graph.temporal_logits.data = graph_state['temporal_logits'].to(graph.temporal_logits.device)

    # Load shared selector state
    if 'shared_selector_state' in graph_state and graph.shared_selector is not None:
        graph.shared_selector.load_state_dict(graph_state['shared_selector_state'])

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


async def execute_graph_on_task(
    graph: LearnableMultiNodeGraph,
    task: str,
    answer: str,
    graph_adj: torch.Tensor,
) -> Dict[str, Any]:
    """
    Execute a single graph topology on a single task.

    Args:
        graph: LearnableMultiNodeGraph with fixed LLM selector
        task: Task text
        answer: Ground truth answer
        graph_adj: Flattened adjacency matrix (N*N,)

    Returns:
        Dict with reward, solved status, and cost
    """
    input_dict = {'task': task}

    # Forward pass with the given graph adjacency
    result = await graph.forward(
        input_dict,
        ground_truth=answer,
        hard=True,  # Use hard selection for evaluation
        is_training=False,
        override_spatial_adj=graph_adj,
    )

    return {
        'reward': result['final_reward'],
        'solved': result['final_solved'],
        'cost': result['total_cost'],
        'prediction': result['final_prediction'],
    }


async def generate_data_for_task(
    graph: LearnableMultiNodeGraph,
    embedder: PositionLLMEmbedder,
    sample: Dict[str, Any],
    num_graphs: int,
    num_nodes: int,
    min_density: float,
    max_density: float,
    top_k: int = 2,
    device: torch.device = None,
) -> List[Dict[str, Any]]:
    """
    Generate labeled data for a single task.

    Args:
        graph: LearnableMultiNodeGraph with fixed LLM selector
        embedder: PositionLLMEmbedder for task embedding
        sample: Dict with 'task' and 'answer'
        num_graphs: Number of random graphs to generate
        num_nodes: Number of nodes in each graph
        min_density: Min edge density for random graphs
        max_density: Max edge density for random graphs
        top_k: Number of top graphs to label as positive
        device: Target device

    Returns:
        List of labeled samples for this task
    """
    task = sample['task']
    answer = sample['answer']

    # Pre-compute task embedding
    task_emb = embedder.get_task_embedding(task, device)

    # Generate random graphs
    graphs = [
        generate_random_graph(num_nodes, min_density, max_density)
        for _ in range(num_graphs)
    ]

    # Execute each graph sequentially
    results = []
    for graph_adj in graphs:
        graph_adj = graph_adj.to(device)
        result = await execute_graph_on_task(graph, task, answer, graph_adj)
        results.append({
            'graph': graph_adj.cpu(),
            'reward': result['reward'],
            'solved': result['solved'],
            'cost': result['cost'],
        })

    # Sort by reward (descending)
    sorted_indices = sorted(range(len(results)), key=lambda i: results[i]['reward'], reverse=True)

    # Label top-k as positive (if reward > 0)
    labeled_data = []
    for rank, idx in enumerate(sorted_indices):
        r = results[idx]
        # Positive if in top-k AND has positive reward
        label = 1 if (rank < top_k and r['reward'] > 0) else 0

        labeled_data.append({
            'task': task,
            'task_emb': task_emb.cpu(),
            'graph': r['graph'],
            'reward': r['reward'],
            'label': label,
            'solved': r['solved'],
            'rank': rank,
            'level': sample.get('level', 'unknown'),  # MATH difficulty level
            'type': sample.get('type', 'unknown'),    # MATH problem type
        })

    return labeled_data


async def process_batch(
    graph: LearnableMultiNodeGraph,
    embedder: PositionLLMEmbedder,
    batch: List[Dict[str, Any]],
    num_graphs: int,
    num_nodes: int,
    min_density: float,
    max_density: float,
    top_k: int,
    device: torch.device,
) -> List[Any]:
    """
    Process a batch of samples in parallel using asyncio.gather.

    Args:
        graph: LearnableMultiNodeGraph with fixed LLM selector
        embedder: PositionLLMEmbedder for task embedding
        batch: List of samples, each with 'task' and 'answer'
        num_graphs: Number of random graphs per task
        num_nodes: Number of nodes in each graph
        min_density: Min edge density for random graphs
        max_density: Max edge density for random graphs
        top_k: Number of top graphs to label as positive
        device: Target device

    Returns:
        List of results (either List[Dict] or Exception for each sample)
    """
    # Use deepcopy to avoid state corruption from shared graph instance
    tasks = [
        generate_data_for_task(
            copy.deepcopy(graph), embedder, sample,
            num_graphs, num_nodes,
            min_density, max_density,
            top_k, device
        )
        for sample in batch
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 2 Data Generation for Graph Classifier Training (MATH)"
    )

    # Stage 1 checkpoint
    parser.add_argument(
        '--stage1_checkpoint',
        type=str,
        required=True,
        help='Path to Stage 1 checkpoint (with trained LLM selector)'
    )

    # Dataset
    parser.add_argument(
        '--dataset_json',
        type=str,
        default='Datasets/MATH/math_train.jsonl',
        help='Path to JSONL file for data generation'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples to process'
    )
    parser.add_argument(
        '--domain',
        type=str,
        default='math',
        help='Domain (mmlu, math, humaneval)'
    )

    # Graph configuration
    parser.add_argument(
        '--multi_node_config',
        type=str,
        default='default',
        help='Path to JSON config or "default"'
    )
    parser.add_argument(
        '--num_graphs_per_task',
        type=int,
        default=5,
        help='Number of random graphs per task'
    )
    parser.add_argument(
        '--graph_min_density',
        type=float,
        default=0.3,
        help='Minimum edge density for random graphs'
    )
    parser.add_argument(
        '--graph_max_density',
        type=float,
        default=1.0,
        help='Maximum edge density for random graphs'
    )
    parser.add_argument(
        '--top_k_positive',
        type=int,
        default=2,
        help='Number of top graphs to label as positive'
    )

    # Model config (must match Stage 1)
    parser.add_argument(
        '--decision_method',
        type=str,
        default='FinalRefer',
        help='Decision node method'
    )
    parser.add_argument(
        '--decision_llm',
        type=str,
        default='gpt-5-mini',
        help='Fixed LLM for decision node'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='FullConnected',
        help='Topology mode (for initialization only)'
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=1,
        help='Number of rounds'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension (must match Stage 1)'
    )

    # LLM exclusion (must match Stage 1)
    parser.add_argument(
        '--exclude_llms',
        type=str,
        nargs='+',
        default=['deepseek-r1-14b'],
        help='LLM names to exclude'
    )

    # Output
    parser.add_argument(
        '--output_file',
        type=str,
        default='result_v2/math/stage2/math_stage2_dataset.pkl',
        help='Output pickle file path'
    )

    # Processing
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Number of tasks to process in parallel (each task runs sequentially for graphs)'
    )
    parser.add_argument(
        '--openrouter_key_variant',
        type=int,
        default=1,
        choices=[1, 2],
        help='OpenRouter API key variant'
    )

    # Skip learning (must match Stage 1)
    parser.add_argument(
        '--enable_skip_learning',
        action='store_true',
        help='Enable skip learning (must match Stage 1)'
    )
    parser.add_argument(
        '--keep_top_k',
        type=int,
        default=3,
        help='Keep top-k roles (must match Stage 1)'
    )
    parser.add_argument(
        '--topK_roles_removed',
        type=int,
        default=0,
        help='Maximum number of roles to skip (only those where skip is argmax)'
    )
    parser.add_argument(
        '--rotation_top_n',
        type=int,
        default=2,
        help='Number of top LLMs to rotate for topN_rotation strategy'
    )

    # Sharing (must match Stage 1)
    parser.add_argument(
        '--share_selector',
        action='store_true',
        help='Share selector (must match Stage 1)'
    )

    # Checkpoint and resume
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Resume from checkpoint file (pickle)'
    )
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=1,
        help='Save checkpoint every N batches (default: 1 = every batch)'
    )

    # Seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


async def main():
    """Main data generation loop."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set OpenRouter key variant
    from HieraMAS.utils.globals import OpenRouterKeyVariant
    OpenRouterKeyVariant.instance().value = args.openrouter_key_variant

    # Load multi-node configs
    if args.multi_node_config == 'default':
        multi_node_configs = DEFAULT_MULTI_NODE_CONFIGS
    else:
        with open(args.multi_node_config, 'r') as f:
            multi_node_configs = json.load(f)

    num_nodes = len(multi_node_configs)

    print("=" * 80)
    print("Stage 2 Data Generation (MATH)")
    print("=" * 80)
    print(f"Stage 1 checkpoint: {args.stage1_checkpoint}")
    print(f"Dataset: {args.dataset_json}")
    print(f"Batch size: {args.batch_size} (parallel samples)")
    print(f"Graphs per task: {args.num_graphs_per_task}")
    print(f"Density range: [{args.graph_min_density}, {args.graph_max_density}]")
    print(f"Top-K positive: {args.top_k_positive}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Output: {args.output_file}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert excluded LLM names to indices
    excluded_llm_indices = None
    if args.exclude_llms:
        excluded_llm_indices = [get_llm_index(name) for name in args.exclude_llms]
        print(f"Excluding LLMs: {args.exclude_llms} (indices: {excluded_llm_indices})")

    # Initialize graph (with NO spatial optimization - we override with random graphs)
    graph = LearnableMultiNodeGraph(
        multi_node_configs=multi_node_configs,
        decision_method=args.decision_method,
        domain=args.domain,
        mode=args.mode,
        num_rounds=args.num_rounds,
        hidden_dim=args.hidden_dim,
        use_llm_for_task_meta=True,
        optimized_spatial=False,  # We override with random graphs
        optimized_temporal=False,
        decision_llm=args.decision_llm,
        share_selector=args.share_selector,
        enable_skip_learning=args.enable_skip_learning,
        keep_top_k=args.keep_top_k,
        excluded_llm_indices=excluded_llm_indices,
    )
    graph.to(device)

    # Load Stage 1 checkpoint
    print(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
    checkpoint = torch.load(args.stage1_checkpoint, map_location=device)
    graph_state = checkpoint.get('graph_state', {})
    load_graph_state(graph, graph_state)
    print("Stage 1 checkpoint loaded successfully")

    # Set to evaluation mode (fixed LLM selector)
    graph.eval()

    # Initialize embedder for task embeddings
    embedder = PositionLLMEmbedder()

    # Load dataset
    dataset = load_dataset(args.dataset_json, args.domain, args.max_samples)
    print(f"Dataset size: {len(dataset)}")

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    all_data = []
    total_positive = 0
    total_negative = 0
    start_batch = 0

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        with open(args.resume_from, 'rb') as f:
            ckpt = pickle.load(f)
        start_batch = ckpt['batch_idx'] + 1
        all_data = ckpt['all_data']
        total_positive = ckpt['total_positive']
        total_negative = ckpt['total_negative']
        print(f"Resumed: batch_idx={ckpt['batch_idx']}, samples={len(all_data)}, "
              f"pos={total_positive}, neg={total_negative}")

    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Processing {len(dataset)} samples in {num_batches} batches (batch_size={batch_size})")

    for batch_idx in tqdm(range(start_batch, num_batches), desc="Generating data",
                          initial=start_batch, total=num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]

        # Process batch in parallel
        batch_results = await process_batch(
            graph=graph,
            embedder=embedder,
            batch=batch,
            num_graphs=args.num_graphs_per_task,
            num_nodes=num_nodes,
            min_density=args.graph_min_density,
            max_density=args.graph_max_density,
            top_k=args.top_k_positive,
            device=device,
        )

        # Collect results, handling exceptions
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"\nError processing sample {start_idx + i}: {result}")
                continue

            labeled_data = result
            all_data.extend(labeled_data)

            # Count labels
            for d in labeled_data:
                if d['label'] == 1:
                    total_positive += 1
                else:
                    total_negative += 1

        # Progress update every 5 batches
        if (batch_idx + 1) % 5 == 0:
            processed = min((batch_idx + 1) * batch_size, len(dataset))
            print(f"\nProcessed {processed}/{len(dataset)} tasks, "
                  f"samples={len(all_data)}, pos={total_positive}, neg={total_negative}")
            print(f"Cost so far: ${Cost.instance().value:.4f}")

        # Save checkpoint after each interval
        if (batch_idx + 1) % args.checkpoint_interval == 0:
            ckpt_file = checkpoint_dir / f"stage2_ckpt_batch_{batch_idx}.pkl"
            with open(ckpt_file, 'wb') as f:
                pickle.dump({
                    'batch_idx': batch_idx,
                    'all_data': all_data,
                    'total_positive': total_positive,
                    'total_negative': total_negative,
                    'cost': Cost.instance().value,
                    'args': vars(args),
                }, f)
            processed = min((batch_idx + 1) * batch_size, len(dataset))
            print(f"Checkpoint saved: {ckpt_file} ({processed}/{len(dataset)} tasks, "
                  f"samples={len(all_data)}, pos={total_positive}, neg={total_negative})")

    # Save dataset
    print(f"\nSaving dataset to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump({
            'data': all_data,
            'metadata': {
                'stage1_checkpoint': args.stage1_checkpoint,
                'dataset_json': args.dataset_json,
                'num_graphs_per_task': args.num_graphs_per_task,
                'graph_density_range': [args.graph_min_density, args.graph_max_density],
                'top_k_positive': args.top_k_positive,
                'num_nodes': num_nodes,
                'total_samples': len(all_data),
                'total_positive': total_positive,
                'total_negative': total_negative,
                'multi_node_configs': multi_node_configs,
            }
        }, f)

    print("=" * 80)
    print("Data Generation Complete!")
    print("=" * 80)
    print(f"Total samples: {len(all_data)}")
    print(f"Positive samples: {total_positive} ({100*total_positive/len(all_data):.1f}%)")
    print(f"Negative samples: {total_negative} ({100*total_negative/len(all_data):.1f}%)")
    print(f"Total cost: ${Cost.instance().value:.4f}")
    print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
