"""
MMLU experiment with LearnableMultiNodeGraph.

This script trains a graph of LearnableMultiNodes using REINFORCE policy gradient.
Each node has its own LLMSelector, and edges between nodes can also be learned.

Usage:
    python experiments/run_mmlu_learnable_graph.py --batch_size 4 --num_iterations 100

    # With custom topology
    python experiments/run_mmlu_learnable_graph.py --mode Chain --num_rounds 1

    # Resume training from checkpoint
    python experiments/run_mmlu_learnable_graph.py --resume_from checkpoints/graph_iter50.pt

    # Evaluate with hard selection
    python experiments/run_mmlu_learnable_graph.py --eval_only --resume_from checkpoints/graph_best.pt
"""

import os
# Must be set BEFORE importing transformers/sentence_transformers
os.environ['TRUST_REMOTE_CODE'] = 'true'
os.environ['HF_HUB_TRUST_REMOTE_CODE'] = '1'

import sys
import copy
import asyncio
import argparse
import json
import time
import random
import math
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.optim import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

from HieraMAS.graph.learnable_multi_node_graph import LearnableMultiNodeGraph
from HieraMAS.graph.random_graph_utils import generate_graph_pool
from HieraMAS.model.llm_selector import TemperatureScheduler
from HieraMAS.prompt.llm_profiles import get_llm_index
from HieraMAS.utils.graph_reward import GraphRewardTracker
from HieraMAS.utils.const import HIERAMAS_ROOT
from HieraMAS.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from HieraMAS.tools.reader.readers import JSONLReader
from termcolor import colored


def get_cosine_warmup_lr(
    iteration: int,
    total_iterations: int,
    lr_start: float,
    lr_end: float,
    warmup_ratio: float
) -> float:
    """
    Cosine warmup: LR increases from lr_start to lr_end over warmup_ratio of total iterations.
    After warmup, LR stays at lr_end.

    Args:
        iteration: Current iteration (0-indexed)
        total_iterations: Total number of training iterations
        lr_start: Initial learning rate
        lr_end: Final learning rate after warmup
        warmup_ratio: Fraction of iterations for warmup (e.g., 0.5 = first 50%)

    Returns:
        Learning rate for current iteration
    """
    warmup_steps = int(total_iterations * warmup_ratio)
    if warmup_steps == 0 or iteration >= warmup_steps:
        return lr_end
    # Cosine warmup: starts slow, accelerates, then slows
    progress = iteration / warmup_steps
    # Cosine goes from 1 to 0, so we use (1 - cos(pi * progress)) / 2 to go from 0 to 1
    cosine_factor = (1 - math.cos(math.pi * progress)) / 2
    return lr_start + (lr_end - lr_start) * cosine_factor


def build_graph_state(graph: LearnableMultiNodeGraph) -> Dict[str, Any]:
    """
    Build complete graph state dict for checkpointing.

    Handles both GCN-based and raw logit-based edge learning,
    as well as shared vs per-node selectors.

    Args:
        graph: LearnableMultiNodeGraph instance

    Returns:
        State dict for saving
    """
    state = {
        'temporal_logits': graph.temporal_logits.data,
        'spatial_masks': graph.spatial_masks.data,
        'temporal_masks': graph.temporal_masks.data,
    }

    # Handle shared vs per-node selectors
    if graph.share_selector and graph.shared_selector is not None:
        state['shared_selector'] = graph.shared_selector.state_dict()
        state['share_selector'] = True
    else:
        state['learnable_nodes'] = {
            role: node.selector.state_dict()
            for role, node in graph.learnable_nodes.items()
        }
        state['share_selector'] = False

    # Save GCN components if using GCN for edges
    if graph.use_gcn_for_edges:
        state['simple_gcn'] = graph.simple_gcn.state_dict()
        state['edge_scorer'] = graph.edge_scorer.state_dict()
        state['use_gcn_for_edges'] = True
    else:
        state['spatial_logits'] = graph.spatial_logits.data
        state['use_gcn_for_edges'] = False

    return state


def load_graph_state(graph: LearnableMultiNodeGraph, graph_state: Dict[str, Any]) -> None:
    """
    Load graph state from checkpoint.

    Handles both GCN-based and raw logit-based edge learning,
    as well as shared vs per-node selectors.

    Args:
        graph: LearnableMultiNodeGraph instance
        graph_state: State dict from checkpoint
    """
    # Handle shared vs per-node selectors
    if graph_state.get('share_selector', False):
        if graph.share_selector and graph.shared_selector is not None:
            graph.shared_selector.load_state_dict(graph_state['shared_selector'])
            print("Loaded shared LLMSelector weights")
    else:
        # Load LLM selectors for each node
        for role, state in graph_state.get('learnable_nodes', {}).items():
            if role in graph.learnable_nodes:
                graph.learnable_nodes[role].selector.load_state_dict(state)

    # Load GCN components if checkpoint used GCN
    if graph_state.get('use_gcn_for_edges', False):
        if graph.use_gcn_for_edges:
            graph.simple_gcn.load_state_dict(graph_state['simple_gcn'])
            graph.edge_scorer.load_state_dict(graph_state['edge_scorer'])
            print("Loaded SimpleGCN and EdgeScorer weights")
        else:
            print("WARNING: Checkpoint has GCN but graph doesn't use GCN!")
    elif 'spatial_logits' in graph_state:
        if hasattr(graph, 'spatial_logits') and not graph.use_gcn_for_edges:
            graph.spatial_logits.data = graph_state['spatial_logits']

    # Load temporal logits
    if 'temporal_logits' in graph_state:
        graph.temporal_logits.data = graph_state['temporal_logits']


# Default multi-node configurations for MMLU
DEFAULT_MULTI_NODE_CONFIGS = [
    {'role': 'Knowlegable Expert', 'num_workers': 6, 'agent_name': 'AnalyzeAgent', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Critic', 'num_workers': 6, 'agent_name': 'AnalyzeAgent', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Mathematician', 'num_workers': 6, 'agent_name': 'AnalyzeAgent', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Psychologist', 'num_workers': 6, 'agent_name': 'AnalyzeAgent', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Historian', 'num_workers': 6, 'agent_name': 'AnalyzeAgent', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Doctor', 'num_workers': 6, 'agent_name': 'AnalyzeAgent', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Lawyer', 'num_workers': 6, 'agent_name': 'AnalyzeAgent', 'aggregator_method': 'FinalMultiAgentAggregator'},
    {'role': 'Economist', 'num_workers': 6, 'agent_name': 'AnalyzeAgent', 'aggregator_method': 'FinalMultiAgentAggregator'},
]


def mmlu_data_process(data: List[Dict]) -> List[Dict]:
    """Process MMLU JSONL data to expected format.

    JSONL format: {"subject": "...", "question": "...", "choices": [...], "answer": 0-3, "correct_answer_letter": "A-D"}
    Expected: {"task": "formatted question with choices", "answer": "A/B/C/D", "subject": "..."}
    """
    processed = []
    choice_labels = ['A', 'B', 'C', 'D']
    for item in data:
        choices = item.get('choices', [])
        formatted_choices = '\n'.join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
        formatted_question = f"{item['question']}\n{formatted_choices}"
        processed.append({
            'task': formatted_question,
            'answer': item.get('correct_answer_letter', choice_labels[item.get('answer', 0)]),
            'subject': item.get('subject', 'unknown'),
        })
    return processed


def load_mmlu_dataset(dataset_json: str, max_samples: int = None) -> list:
    """Load MMLU dataset from JSONL file."""
    print(f"Loading dataset: {dataset_json}")
    data = JSONLReader.parse_file(dataset_json)
    samples = mmlu_data_process(data)
    if max_samples:
        samples = samples[:max_samples]
    return samples



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MMLU experiment with LearnableMultiNodeGraph"
    )

    # Dataset
    parser.add_argument(
        '--dataset_json',
        type=str,
        default='Datasets/MMLU/train.jsonl',
        help='Path to JSONL file for training'
    )
    parser.add_argument(
        '--valid_dataset_json',
        type=str,
        default='Datasets/MMLU/valid.jsonl',
        help='Path to JSONL file for validation'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples to use (for testing)'
    )

    # Graph configuration
    parser.add_argument(
        '--multi_node_config',
        type=str,
        default='default',
        help='Path to JSON config file or "default"'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='FullConnected',
        choices=['Chain', 'FullConnected', 'Star', 'Debate', 'DirectAnswer', 'Layered', 'Mesh'],
        help='Topology mode'
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=1,
        help='Number of iteration rounds'
    )
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
        '--domain',
        type=str,
        default='mmlu',
        help='Domain for prompts'
    )

    # Model configuration
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension for selector networks'
    )
    parser.add_argument(
        '--initial_temp',
        type=float,
        default=10.0,
        help='Initial Gumbel-Softmax temperature'
    )
    parser.add_argument(
        '--min_temp',
        type=float,
        default=0.8,
        help='Minimum temperature'
    )
    parser.add_argument(
        '--temp_decay',
        type=float,
        default=0.95,
        help='Temperature decay rate'
    )
    parser.add_argument(
        '--edge_temp',
        type=float,
        default=10.0,
        help='Edge sampling temperature'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=10,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-2,
        help='Learning rate (legacy, use lr_start/lr_end for scheduler)'
    )
    parser.add_argument(
        '--lr_start',
        type=float,
        default=2e-3,
        help='Initial learning rate for warmup'
    )
    parser.add_argument(
        '--lr_end',
        type=float,
        default=2e-3,
        help='Final learning rate after warmup'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.5,
        help='Fraction of iterations for LR warmup (0.5 = first 50%%)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Reward mixing coefficient (0=only final, 1=only node)'
    )
    parser.add_argument(
        '--cost_sensitivity',
        type=float,
        default=300.0,
        help='Cost sensitivity in reward function'
    )
    parser.add_argument(
        '--optimized_spatial',
        action='store_true',
        help='Learn spatial edges'
    )
    parser.add_argument(
        '--optimized_temporal',
        action='store_true',
        help='Learn temporal edges'
    )
    parser.add_argument(
        '--share_selector',
        action='store_true',
        help='Share a single LLMSelector across all MultiNodes'
    )
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=1.0,
        help='Entropy regularization coefficient (0.8 recommended). '
             'Higher values encourage more exploration and prevent policy collapse.'
    )
    parser.add_argument(
        '--hoyer_coef',
        type=float,
        default=0.8,
        help='Hoyer sparsity regularization coefficient (0.8 recommended). '
             'Encourages sparse edge configurations (few high-probability edges, most near-zero). '
             'Set to 0.0 to disable sparsity regularization.'
    )

    # Skip learning parameters
    parser.add_argument(
        '--enable_skip_learning',
        action='store_true',
        help='Enable learning to skip roles based on relative performance'
    )
    parser.add_argument(
        '--keep_top_k',
        type=int,
        default=3,
        help='Number of top-performing roles to keep (rest will be learned to skip)'
    )
    parser.add_argument(
        '--skip_coef',
        type=float,
        default=0.5,
        help='Skip loss coefficient (weight for BCE skip loss)'
    )
    parser.add_argument(
        '--skip_warmup_iterations',
        type=int,
        default=10,
        help='Number of iterations before enabling skip learning loss'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=3,
        help='Save checkpoint every N iterations'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    # Evaluation
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only evaluate (no training)'
    )

    # Validation during training
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=8,
        help='Run validation every N training iterations'
    )
    parser.add_argument(
        '--max_valid_samples',
        type=int,
        default=None,
        help='Maximum validation samples to use'
    )

    # Results
    parser.add_argument(
        '--result_file',
        type=str,
        default=None,
        help='Custom result file path'
    )

    # LLM exclusion
    parser.add_argument(
        '--exclude_llms',
        type=str,
        nargs='+',
        default=['deepseek-r1-14b'],
        help='LLM names to exclude from selection (default: deepseek-r1-14b)'
    )

    # OpenRouter API key selection
    parser.add_argument(
        '--openrouter_key_variant',
        type=int,
        default=1,
        choices=[1, 2],
        help='OpenRouter API key variant: 1=OPENROUTER_API_KEY (default), 2=OPENROUTER_API_KEY_2'
    )

    # Parallel processing
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=None,
        help='Max concurrent API calls in batch (default: unlimited/parallel). Set to 1 for sequential.'
    )

    # Random graph pool (Stage 1 training)
    parser.add_argument(
        '--use_random_graph_pool',
        action='store_true',
        help='Use pre-generated random graphs instead of learning edges (Stage 1 training)'
    )
    parser.add_argument(
        '--num_graph_candidates',
        type=int,
        default=100,
        help='Number of random graph candidates to pre-generate (max 100)'
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
        default=0.7,
        help='Maximum edge density for random graphs'
    )

    return parser.parse_args()


def load_multi_node_configs(config_path: str) -> List[Dict[str, Any]]:
    """Load multi-node configurations from file or use defaults."""
    if config_path == 'default':
        return DEFAULT_MULTI_NODE_CONFIGS

    with open(config_path, 'r') as f:
        return json.load(f)


async def train_step(
    graph: LearnableMultiNodeGraph,
    batch: List[Dict[str, Any]],
    entropy_coef: float = 0.8,
    hoyer_coef: float = 0.8,
    skip_coef: float = 0.0,
    max_concurrent: int = None,
    graph_pool: List[torch.Tensor] = None,
) -> tuple:
    """
    Execute one training step on a batch IN PARALLEL.

    Args:
        graph: LearnableMultiNodeGraph instance
        batch: List of samples with 'task' and 'answer' keys
        entropy_coef: Entropy regularization coefficient
        hoyer_coef: Hoyer sparsity regularization coefficient
        skip_coef: Skip learning loss coefficient (0.0 disables skip loss)
        max_concurrent: Max concurrent API calls (None = unlimited/parallel)
        graph_pool: List of pre-generated graph adjacency tensors. If provided,
                    each sample will use a randomly selected graph from the pool.

    Returns:
        batch_losses: List of loss tensors
        batch_results: List of forward results
    """
    async def process_sample(sample, graph_instance):
        input_dict = {'task': sample['task']}

        # Randomly select a graph from pool for THIS SAMPLE (per-sample selection)
        selected_graph = None
        if graph_pool is not None:
            selected_graph = random.choice(graph_pool)

        # Forward pass with per-sample graph override
        result = await graph_instance.forward(
            input_dict,
            ground_truth=sample['answer'],
            hard=False,
            override_spatial_adj=selected_graph,
        )

        # Compute loss with entropy and skip regularization
        # When using graph_pool, hoyer_coef is ignored (no edge learning)
        effective_hoyer_coef = 0.0 if graph_pool is not None else hoyer_coef
        loss = graph_instance.compute_loss(
            result,
            entropy_coef=entropy_coef,
            hoyer_coef=effective_hoyer_coef,
            skip_coef=skip_coef,
        )
        return loss, result

    # Run all samples in parallel (with optional concurrency limit)
    # No deepcopy needed: mutable state removed from LearnableMultiNode
    # (tokens now returned in result dict instead of stored in self._last_*)
    if max_concurrent is not None and max_concurrent > 0:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_process(sample, graph_instance):
            async with semaphore:
                return await process_sample(sample, graph_instance)

        tasks = [limited_process(sample, graph) for sample in batch]
    else:
        tasks = [process_sample(sample, graph) for sample in batch]

    results = await asyncio.gather(*tasks)

    batch_losses = [r[0] for r in results]
    batch_results = [r[1] for r in results]

    return batch_losses, batch_results


async def eval_step(
    graph: LearnableMultiNodeGraph,
    batch: List[Dict[str, Any]],
    max_concurrent: int = None,
) -> List[Dict[str, Any]]:
    """
    Execute one evaluation step on a batch IN PARALLEL (hard selection).

    Uses deep copies to prevent state corruption from concurrent access.

    Args:
        graph: LearnableMultiNodeGraph instance
        batch: List of samples with 'task' and 'answer' keys
        max_concurrent: Max concurrent API calls (None = unlimited/parallel)

    Returns:
        List of forward results
    """
    # Create deep copies for each sample to prevent state corruption
    graph_copies = [copy.deepcopy(graph) for _ in batch]

    async def process_sample(sample, graph_copy):
        input_dict = {'task': sample['task']}

        # Forward pass (hard selection)
        result = await graph_copy.forward(
            input_dict,
            ground_truth=sample['answer'],
            hard=True,
        )
        return result

    # Run all samples in parallel (with optional concurrency limit)
    if max_concurrent is not None and max_concurrent > 0:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_process(sample, graph_copy):
            async with semaphore:
                return await process_sample(sample, graph_copy)

        tasks = [limited_process(sample, gc) for sample, gc in zip(batch, graph_copies)]
    else:
        tasks = [process_sample(sample, gc) for sample, gc in zip(batch, graph_copies)]

    results = await asyncio.gather(*tasks)

    return list(results)


async def run_validation(
    graph: LearnableMultiNodeGraph,
    valid_dataset: List[Dict[str, Any]],
    batch_size: int,
    max_concurrent: int = None,
) -> Dict[str, Any]:
    """
    Run validation on the entire validation dataset.

    Args:
        graph: LearnableMultiNodeGraph instance
        valid_dataset: List of validation samples
        batch_size: Batch size for processing
        max_concurrent: Max concurrent API calls (None = unlimited/parallel)

    Returns:
        Dictionary with validation metrics including solve_rate, mean_reward, total_cost
    """
    graph.eval()
    valid_tracker = GraphRewardTracker()

    num_batches = (len(valid_dataset) + batch_size - 1) // batch_size

    for i_batch in range(num_batches):
        start_idx = i_batch * batch_size
        batch = valid_dataset[start_idx:start_idx + batch_size]

        results = await eval_step(graph, batch, max_concurrent=max_concurrent)

        for result in results:
            valid_tracker.add(result)

    graph.train()  # Switch back to training mode

    summary = valid_tracker.summary()
    return {
        'solve_rate': summary['final'].get('solve_rate', 0.0),
        'mean_reward': summary['final'].get('mean_reward', 0.0),
        'total_cost': summary['final'].get('total_cost', 0.0),
        'num_samples': len(valid_dataset),
    }


async def main():
    """Main training loop."""
    args = parse_args()

    # Set OpenRouter key variant (must be done before any OpenRouter API calls)
    from HieraMAS.utils.globals import OpenRouterKeyVariant
    OpenRouterKeyVariant.instance().value = args.openrouter_key_variant

    # Load multi-node configs
    multi_node_configs = load_multi_node_configs(args.multi_node_config)

    print("=" * 80)
    print("LearnableMultiNodeGraph Experiment - MMLU")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Num Rounds: {args.num_rounds}")
    print(f"Alpha: {args.alpha}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr_start} -> {args.lr_end} (warmup {args.warmup_ratio*100:.0f}%)")
    print(f"Optimized Spatial: {args.optimized_spatial}")
    print(f"Share Selector: {args.share_selector}")
    print(f"Eval only: {args.eval_only}")
    print(f"Skip Learning: {args.enable_skip_learning}")
    if args.enable_skip_learning:
        print(f"  Keep Top-K: {args.keep_top_k}")
        print(f"  Skip Coef: {args.skip_coef}")
        print(f"  Skip Warmup Iterations: {args.skip_warmup_iterations}")
    print(f"Random Graph Pool: {args.use_random_graph_pool}")
    if args.use_random_graph_pool:
        print(f"  Num Candidates: {args.num_graph_candidates}")
        print(f"  Density Range: [{args.graph_min_density}, {args.graph_max_density}]")
    print(f"Train dataset: {args.dataset_json}")
    parallel_mode = "unlimited" if args.max_concurrent is None else str(args.max_concurrent)
    print(f"Parallel processing: max_concurrent={parallel_mode}")
    print(f"Nodes:")
    for config in multi_node_configs:
        print(f"  - {config['role']}: {config.get('num_workers', 3)} workers")
    print("=" * 80)

    # Setup directories
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S")
    Time.instance().value = current_time

    # Determine result_dir from --result_file or use default
    if args.result_file:
        result_file = Path(args.result_file)
        result_dir = result_file.parent
    else:
        result_dir = Path(f"{HIERAMAS_ROOT}/result/mmlu_learnable_graph")
        result_file = result_dir / f"graph_{args.mode}_{current_time}.json"

    # Create directory structure
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = result_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    test_dir = result_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Initialize graph
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert excluded LLM names to indices
    excluded_llm_indices = None
    if args.exclude_llms:
        excluded_llm_indices = [get_llm_index(name) for name in args.exclude_llms]
        print(f"Excluding LLMs: {args.exclude_llms} (indices: {excluded_llm_indices})")

    # When using random graph pool, disable spatial edge learning
    effective_optimized_spatial = False if args.use_random_graph_pool else args.optimized_spatial

    graph = LearnableMultiNodeGraph(
        multi_node_configs=multi_node_configs,
        decision_method=args.decision_method,
        domain=args.domain,
        mode=args.mode,
        num_rounds=args.num_rounds,
        hidden_dim=args.hidden_dim,
        initial_temperature=args.initial_temp,
        edge_temperature=args.edge_temp,
        use_llm_for_task_meta=True,
        alpha=args.alpha,
        optimized_spatial=effective_optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        decision_llm=args.decision_llm,
        cost_sensitivity=args.cost_sensitivity,
        share_selector=args.share_selector,
        enable_skip_learning=args.enable_skip_learning,
        keep_top_k=args.keep_top_k,
        excluded_llm_indices=excluded_llm_indices,
    )
    graph.to(device)

    # Generate random graph pool if enabled (Stage 1 training)
    graph_pool = None
    if args.use_random_graph_pool:
        num_nodes = len(multi_node_configs)
        graph_pool = generate_graph_pool(
            num_graphs=min(args.num_graph_candidates, 100),
            num_nodes=num_nodes,
            min_density=args.graph_min_density,
            max_density=args.graph_max_density,
        )
        print(f"Generated {len(graph_pool)} random graph candidates (nodes={num_nodes})")

    print(graph.get_config_summary())

    # Collect trainable parameters
    # When using random graph pool, exclude edge-related parameters
    if args.use_random_graph_pool:
        # Only optimize LLM selector parameters (not edge parameters)
        all_params = []
        if graph.shared_selector is not None:
            all_params.extend(graph.shared_selector.parameters())
        else:
            for node in graph.learnable_nodes.values():
                if hasattr(node, 'selector') and node.selector is not None:
                    all_params.extend(node.selector.parameters())
        # Note: Skip logits from aggregators would be added here if skip learning is enabled
        # but they're already included in graph.parameters() and don't involve edge learning
        print(f"Using random graph pool: optimizing LLM selector only")
    else:
        all_params = list(graph.parameters())
    print(f"Total trainable parameters: {sum(p.numel() for p in all_params if p.requires_grad)}")

    # Optimizer (starts with lr_start, will warmup to lr_end)
    optimizer = Adam(all_params, lr=args.lr_start, weight_decay=1e-5)

    # Temperature scheduler
    temp_scheduler = TemperatureScheduler(
        initial_temp=args.initial_temp,
        min_temp=args.min_temp,
        schedule='cosine',
        total_steps=args.num_iterations
    )

    # Resume from checkpoint
    start_iter = 0
    best_solve_rate = 0.0
    best_valid_solve_rate = 0.0
    resumed_checkpoint = None
    if args.resume_from:
        print(f"Loading checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)

        # Load graph state using helper function
        graph_state = checkpoint.get('graph_state', {})
        load_graph_state(graph, graph_state)

        if 'optimizer_state' in checkpoint and not args.eval_only:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        start_iter = checkpoint.get('iteration', 0) + 1
        best_solve_rate = checkpoint.get('best_solve_rate', 0.0)
        best_valid_solve_rate = checkpoint.get('best_valid_solve_rate', 0.0)

        if 'temp_scheduler' in checkpoint:
            temp_scheduler.load_state_dict(checkpoint['temp_scheduler'])

        resumed_checkpoint = checkpoint
        print(f"Resumed from iteration {start_iter}")

    # Load dataset
    dataset = load_mmlu_dataset(args.dataset_json, args.max_samples)
    print(f"Dataset size: {len(dataset)}")

    # Load validation dataset
    valid_dataset = None
    if args.valid_dataset_json:
        valid_dataset = load_mmlu_dataset(args.valid_dataset_json, args.max_valid_samples)
        print(f"Validation dataset size: {len(valid_dataset)}")

    # Create best model directory
    best_model_dir = checkpoint_dir / "bestModel"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation only mode
    if args.eval_only:
        print("\nRunning evaluation only...")
        graph.eval()
        reward_tracker = GraphRewardTracker()
        all_results = []

        num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

        for i_batch in range(num_batches):
            start_idx = i_batch * args.batch_size
            batch = dataset[start_idx:start_idx + args.batch_size]

            results = await eval_step(graph, batch, max_concurrent=args.max_concurrent)

            for result, sample in zip(results, batch):
                reward_tracker.add(result)
                all_results.append({
                    'task': sample['task'],
                    'answer': sample['answer'],
                    'subject': sample.get('subject', 'unknown'),
                    'final_prediction': result['final_prediction'],
                    'final_solved': result['final_solved'],
                    'total_cost': result['total_cost'],
                    'node_results': {
                        role: {
                            'prediction': r['prediction'],
                            'solved': r['solved'],
                            'cost': r['cost'],
                            'llm_selection': r['llm_selection'],
                        }
                        for role, r in result['node_results'].items()
                    },
                })

            if (i_batch + 1) % 5 == 0:
                summary = reward_tracker.summary()
                print(f"Batch {i_batch+1}/{num_batches}: "
                      f"solve_rate={summary['final'].get('solve_rate', 0.0):.4f}")

        # Final summary
        print(reward_tracker.format_final_summary())

        # Save results
        summary = reward_tracker.summary()
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'config': {
                    'mode': args.mode,
                    'num_rounds': args.num_rounds,
                    'alpha': args.alpha,
                },
                'results': all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {result_file}")
        return

    # Training mode
    print("\nStarting training...")
    graph.train()
    reward_tracker = GraphRewardTracker()

    # Restore reward tracker state if resuming
    if resumed_checkpoint and 'reward_tracker' in resumed_checkpoint:
        reward_tracker.load_state_dict(resumed_checkpoint['reward_tracker'])
        print(f"Restored reward tracker: {len(reward_tracker)} samples")

    all_results = []

    # Shuffle dataset before training
    random.shuffle(dataset)
    print(f"Shuffled training dataset ({len(dataset)} samples)")

    # Calculate epoch size for re-shuffling
    epoch_size = len(dataset) // args.batch_size

    for iteration in range(start_iter, args.num_iterations):
        # Re-shuffle at the start of each epoch
        if iteration > 0 and iteration % epoch_size == 0:
            random.shuffle(dataset)
            print(f"Re-shuffled dataset at iteration {iteration} (epoch {iteration // epoch_size})")

        # Get batch
        batch_start = (iteration * args.batch_size) % len(dataset)
        batch = dataset[batch_start:batch_start + args.batch_size]

        # Compute effective skip_coef (warmup)
        if args.enable_skip_learning and iteration >= args.skip_warmup_iterations:
            effective_skip_coef = args.skip_coef
        else:
            effective_skip_coef = 0.0

        # Training step (parallel batch processing)
        # Pass graph_pool for per-sample random graph selection (Stage 1 training)
        batch_losses, batch_results = await train_step(
            graph,
            batch,
            entropy_coef=args.entropy_coef,
            hoyer_coef=args.hoyer_coef,
            skip_coef=effective_skip_coef,
            max_concurrent=args.max_concurrent,
            graph_pool=graph_pool,
        )

        # Record results
        for result in batch_results:
            reward_tracker.add(result)

        # Compute total loss (for logging only)
        total_loss = torch.stack(batch_losses).mean()
        print(colored(f"Batch loss: {total_loss.item():.4f}", "red"))

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Track gradients for all modules
        print(colored("\n=== Gradient Tracking ===", "cyan"))

        # 1. Check shared selector (if exists)
        if graph.shared_selector is not None:
            print(colored("Shared LLMSelector:", "yellow"))
            for name, param in graph.shared_selector.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"  {name}: grad_norm={grad_norm:.6f}, requires_grad={param.requires_grad}")
                else:
                    print(colored(f"  {name}: NO GRAD, requires_grad={param.requires_grad}", "red"))

        # 2. Check GCN model
        if hasattr(graph, 'simple_gcn') and graph.simple_gcn is not None:
            print(colored("SimpleGCN:", "yellow"))
            for name, param in graph.simple_gcn.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"  {name}: grad_norm={grad_norm:.6f}, requires_grad={param.requires_grad}")
                else:
                    print(colored(f"  {name}: NO GRAD, requires_grad={param.requires_grad}", "red"))

        # 3. Check edge scorer
        if hasattr(graph, 'edge_scorer') and graph.edge_scorer is not None:
            print(colored("EdgeScorer:", "yellow"))
            for name, param in graph.edge_scorer.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"  {name}: grad_norm={grad_norm:.6f}, requires_grad={param.requires_grad}")
                else:
                    print(colored(f"  {name}: NO GRAD, requires_grad={param.requires_grad}", "red"))

        # 4. Check spatial/temporal logits
        print(colored("Edge Logits:", "yellow"))
        if graph.spatial_logits.grad is not None:
            print(f"  spatial_logits: grad_norm={graph.spatial_logits.grad.norm().item():.6f}, requires_grad={graph.spatial_logits.requires_grad}")
        else:
            print(colored(f"  spatial_logits: NO GRAD, requires_grad={graph.spatial_logits.requires_grad}", "red"))

        if graph.temporal_logits.grad is not None:
            print(f"  temporal_logits: grad_norm={graph.temporal_logits.grad.norm().item():.6f}, requires_grad={graph.temporal_logits.requires_grad}")
        else:
            print(colored(f"  temporal_logits: NO GRAD, requires_grad={graph.temporal_logits.requires_grad}", "red"))

        # 5. Check learnable nodes' selectors (if not shared)
        if not graph.share_selector:
            print(colored("Per-node LLMSelectors:", "yellow"))
            for role, node in graph.learnable_nodes.items():
                if hasattr(node, 'selector') and node.selector is not None:
                    has_grad = any(p.grad is not None for p in node.selector.parameters())
                    total_grad_norm = sum(p.grad.norm().item() for p in node.selector.parameters() if p.grad is not None)
                    if has_grad:
                        print(f"  {role}: total_grad_norm={total_grad_norm:.6f}")
                    else:
                        print(colored(f"  {role}: NO GRAD", "red"))

        print(colored("=== End Gradient Tracking ===\n", "cyan"))

        torch.nn.utils.clip_grad_norm_(graph.parameters(), max_norm=1.0)
        optimizer.step()

        # Learning rate warmup
        current_lr = get_cosine_warmup_lr(
            iteration, args.num_iterations,
            args.lr_start, args.lr_end, args.warmup_ratio
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Temperature annealing
        temp_scheduler.step()
        graph.set_temperature(temp_scheduler.get_temperature())
        graph.set_edge_temperature(temp_scheduler.get_temperature())
        # Logging
        if (iteration + 1) % 2 == 0:
            print(f"\033[91m{reward_tracker.format_progress(iteration + 1, args.num_iterations, total_loss.item(), temp_scheduler.get_temperature())} | lr={current_lr:.5f}\033[0m")

            # Log skip statistics if skip learning is enabled
            if args.enable_skip_learning:
                skip_rates = reward_tracker.get_skip_rates()
                if skip_rates:
                    skip_status = "ACTIVE" if effective_skip_coef > 0 else f"WARMUP ({iteration}/{args.skip_warmup_iterations})"
                    print(f"\n=== Skip Learning [{skip_status}] ===")
                    for role, rate in skip_rates.items():
                        print(f"  {role}: skip_rate={rate:.2%}")
                    print(f"  Total skip samples: {reward_tracker._skip_samples}")

            # Edge distribution monitoring
            if graph.optimized_spatial and graph.use_gcn_for_edges:
                with torch.no_grad():
                    task = batch[0]['task'] if batch else ""
                    spatial_logits = graph.compute_spatial_logits(task)
                    probs = torch.sigmoid(spatial_logits / graph.edge_temperature)
                    masked_probs = (probs * graph.spatial_masks).cpu().numpy()
                    active_probs = masked_probs[masked_probs > 0]

                    if len(active_probs) > 0:
                        print(f"\n=== Edge Distribution ===")
                        print(f"Mean: {active_probs.mean():.4f}, Std: {active_probs.std():.4f}")
                        print(f"Sparse (p<0.2): {(active_probs < 0.2).sum()}/{len(active_probs)}")
                        print(f"Active (p>0.7): {(active_probs > 0.7).sum()}/{len(active_probs)}")

        # Save results every iteration
        summary = reward_tracker.summary()
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'iteration': iteration + 1,
                'summary': summary,
                'args': vars(args),
                'total_time': Time.instance().value,
                'total_cost': Cost.instance().value,
                'best_solve_rate': best_solve_rate,
                'best_valid_solve_rate': best_valid_solve_rate,
                'config': {
                    'mode': args.mode,
                    'num_rounds': args.num_rounds,
                    'alpha': args.alpha,
                    'multi_node_configs': multi_node_configs,
                },
            }, f, indent=2, ensure_ascii=False)

        # Checkpoint
        if (iteration + 1) % args.checkpoint_interval == 0:
            # Build graph state using helper function
            graph_state = build_graph_state(graph)

            ckpt_path = checkpoint_dir / f"graph_iter{iteration+1}.pt"
            torch.save({
                'iteration': iteration,
                'graph_state': graph_state,
                'optimizer_state': optimizer.state_dict(),
                'temp_scheduler': temp_scheduler.state_dict(),
                'reward_tracker': reward_tracker.state_dict(),
                'config': {
                    'mode': args.mode,
                    'num_rounds': args.num_rounds,
                    'alpha': args.alpha,
                    'multi_node_configs': multi_node_configs,
                },
                'best_solve_rate': best_solve_rate,
                'best_valid_solve_rate': best_valid_solve_rate,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # Save best model
            summary = reward_tracker.summary()
            current_solve_rate = summary['final'].get('solve_rate', 0.0)
            if current_solve_rate > best_solve_rate:
                best_solve_rate = current_solve_rate
                best_path = checkpoint_dir / "graph_best.pt"
                torch.save({
                    'iteration': iteration,
                    'graph_state': graph_state,
                    'config': {
                        'mode': args.mode,
                        'num_rounds': args.num_rounds,
                        'alpha': args.alpha,
                        'multi_node_configs': multi_node_configs,
                    },
                    'solve_rate': best_solve_rate,
                }, best_path)
                print(f"New best model! solve_rate={best_solve_rate:.4f}")

        # Periodic validation
        if valid_dataset is not None and (iteration + 1) % args.eval_steps == 0:
            print(f"\n{'='*40}")
            print(f"Running validation at iteration {iteration + 1}...")

            valid_metrics = await run_validation(graph, valid_dataset, args.batch_size, max_concurrent=args.max_concurrent)

            print(f"Validation Results:")
            print(f"  Solve Rate: {valid_metrics['solve_rate']:.4f}")
            print(f"  Mean Reward: {valid_metrics['mean_reward']:.4f}")
            print(f"  Total Cost: ${valid_metrics['total_cost']:.4f}")

            # Save best model based on validation solve rate
            if valid_metrics['solve_rate'] > best_valid_solve_rate:
                best_valid_solve_rate = valid_metrics['solve_rate']

                # Build graph state using helper function
                graph_state = build_graph_state(graph)

                best_valid_path = best_model_dir / "best_valid_model.pt"
                torch.save({
                    'iteration': iteration,
                    'graph_state': graph_state,
                    'optimizer_state': optimizer.state_dict(),
                    'temp_scheduler': temp_scheduler.state_dict(),
                    'config': {
                        'mode': args.mode,
                        'num_rounds': args.num_rounds,
                        'alpha': args.alpha,
                        'multi_node_configs': multi_node_configs,
                    },
                    'valid_solve_rate': best_valid_solve_rate,
                    'valid_metrics': valid_metrics,
                }, best_valid_path)
                print(colored(f"New best validation model! solve_rate={best_valid_solve_rate:.4f}", "green"))

            print(f"{'='*40}\n")

    # Final summary
    print(reward_tracker.format_final_summary())
    print(f"Best solve rate: {best_solve_rate:.4f}")
    print(f"Best validation solve rate: {best_valid_solve_rate:.4f}")
    print(f"Total cost: ${Cost.instance().value:.4f}")

    # Save training summary to result file
    summary = reward_tracker.summary()
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'args': vars(args),
            'total_time': Time.instance().value,
            'total_cost': Cost.instance().value,
            'best_solve_rate': best_solve_rate,
            'best_valid_solve_rate': best_valid_solve_rate,
            'config': {
                'mode': args.mode,
                'num_rounds': args.num_rounds,
                'alpha': args.alpha,
                'multi_node_configs': multi_node_configs,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"Training results saved to: {result_file}")

    # Save final checkpoint
    graph_state = build_graph_state(graph)

    final_path = checkpoint_dir / "graph_final.pt"
    torch.save({
        'iteration': args.num_iterations - 1,
        'graph_state': graph_state,
        'optimizer_state': optimizer.state_dict(),
        'temp_scheduler': temp_scheduler.state_dict(),
        'config': {
            'mode': args.mode,
            'num_rounds': args.num_rounds,
            'alpha': args.alpha,
            'multi_node_configs': multi_node_configs,
        },
        'best_solve_rate': best_solve_rate,
        'best_valid_solve_rate': best_valid_solve_rate,
    }, final_path)
    print(f"Saved final checkpoint: {final_path}")


if __name__ == '__main__':
    asyncio.run(main())
