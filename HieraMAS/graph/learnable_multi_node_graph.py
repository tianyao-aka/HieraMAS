"""
LearnableMultiNodeGraph: Graph of LearnableMultiNodes with learnable topology.

This module combines learnable LLM selection (per-node LLMSelector) with
learnable graph structure (edge logits), enabling end-to-end optimization
of both "which LLM to use" and "how agents should collaborate."
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import deque
import shortuuid
import numpy as np

from HieraMAS.graph.learnable_multi_node import LearnableMultiNode
from HieraMAS.graph.multi_node_graph import get_mode_masks
from HieraMAS.agents.agent_registry import AgentRegistry
from HieraMAS.model.llm_selector import LLMSelector
from HieraMAS.prompt.llm_profiles import LLM_OPTIONS
from HieraMAS.utils.globals import Cost
from HieraMAS.utils.reward import reward_v2
from HieraMAS.utils.graph_reward import compute_mixed_reward
from HieraMAS.gnn.simple_gcn import SimpleGCN, EdgeScorer, soft_clamp
from HieraMAS.llm.profile_embedding import get_sentence_embedding
from HieraMAS.utils.task_cache import TaskMetaCache
from torch_geometric.utils import dense_to_sparse

from Datasets.MATH import math_get_predict


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    if ans is None:
        return ""
    ans = str(ans).strip()
    ans = ans.replace(" ", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\!", "").replace("\\,", "")
    return ans


def check_answer(predict: str, truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    try:
        return float(predict) == float(truth)
    except (ValueError, TypeError):
        return normalize_answer(predict) == normalize_answer(truth)


def humaneval_extract_code(response: str) -> str:
    """Extract Python code from LLM response.

    Handles various markdown code block formats:
    - ```python ... ```
    - ``` ... ```
    - Raw code without markdown
    """
    code = response.strip()
    if code.startswith("```python"):
        code = code.split("```python", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
    elif code.startswith("```"):
        code = code.split("```", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
    return code.strip()


def extract_function_name(code: str) -> str:
    """Extract the main function name from code.

    Args:
        code: Python code containing function definition

    Returns:
        Function name or 'solution' as fallback
    """
    import re
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    return "solution"


def humaneval_check_code(code: str, test: str, timeout: int = 150) -> bool:
    """Check if code passes the test using PyExecutor.

    The test string contains check(candidate) function definition.
    For evalplus tests, there may also be a check_plus(candidate) function.
    We execute the code and call these functions with the actual function name.

    Args:
        code: Python code to evaluate
        test: Test case string with check() and optionally check_plus() function definitions
        timeout: Execution timeout in seconds (for ALL tests combined)

    Returns:
        True if code passes all tests, False otherwise
    """
    from HieraMAS.tools.coding.executor_utils import function_with_timeout

    try:
        # Extract the function name from the generated code
        func_name = extract_function_name(code)

        # Check if test has check_plus function
        has_check_plus = 'def check_plus' in test

        # Build full execution code:
        # 1. Import typing
        # 2. User's code (function definition)
        # 3. Test definitions (check and optionally check_plus functions)
        # 4. Call check() and check_plus() with the function name (if exists)
        if has_check_plus:
            full_code = f"""from typing import *

{code}

{test}

check({func_name})
check_plus({func_name})
"""
        else:
            full_code = f"""from typing import *

{code}

{test}

check({func_name})
"""
        # Execute with timeout
        function_with_timeout(exec, (full_code, {}), timeout)
        return True
    except Exception:
        return False


def get_answer_extractor(domain: str):
    """Get domain-specific answer extraction function.

    For MMLU and GSM8K, we pass the full output to the LLM judge
    which can handle CoT outputs with analysis before the answer.
    """
    if domain == 'mmlu':
        # No extraction - pass full output to LLM judge
        return lambda x: x.strip() if isinstance(x, str) else str(x).strip()
    elif domain in ('gsm8k', 'math'):
        # No extraction - pass full output to LLM judge
        return lambda x: x.strip() if isinstance(x, str) else str(x).strip()
    elif domain == 'humaneval':
        return humaneval_extract_code  # Keep - extracts code blocks
    else:
        return lambda x: str(x).strip()


async def llm_judge_math_answer(prediction: str, ground_truth: str) -> bool:
    """Use gpt-4o-mini to judge if prediction matches ground truth.

    Handles CoT outputs where the model shows step-by-step reasoning.

    Args:
        prediction: Model's predicted answer
        ground_truth: Expected correct answer

    Returns:
        True if answers are mathematically equivalent, False otherwise
    """
    from HieraMAS.llm.gpt_chat import achat

    prompt = f"""Analyze the following mathematical response and determine if the final answer matches the ground truth.

Response:
{prediction}

Ground Truth: {ground_truth}

The response may contain step-by-step reasoning before the final answer.
Look for the final numerical answer in the response.
Are the final answer and ground truth mathematically equivalent? Reply with only "yes" or "no"."""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await achat("gpt-4o-mini", messages)
        answer = response.strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


async def llm_judge_mmlu_answer(prediction: str, ground_truth: str) -> bool:
    """Use gpt-4o-mini to judge if MMLU prediction matches ground truth.

    Handles CoT outputs where the model analyzes options before giving final answer.

    Args:
        prediction: Model's predicted answer (may contain explanation)
        ground_truth: Expected correct answer (A, B, C, or D)

    Returns:
        True if prediction indicates the correct answer, False otherwise
    """
    from HieraMAS.llm.gpt_chat import achat

    prompt = f"""Analyze the following response and determine if the final answer is "{ground_truth}".

Response:
{prediction}

Expected Answer: {ground_truth}

The response may contain analysis/reasoning before the final answer.
Look for the final answer choice (A, B, C, or D) that the response concludes with.
Does the response indicate the final answer is {ground_truth}? Reply with only "yes" or "no"."""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await achat("gpt-4o-mini", messages)
        answer = response.strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


def get_answer_checker(domain: str):
    """Get domain-specific answer checking function.

    All returned functions are async for consistency.
    For math/gsm8k domains, uses LLM-based evaluation with gpt-4o-mini.
    For mmlu domain, uses LLM-based evaluation directly.
    """
    if domain == 'mmlu':
        return llm_judge_mmlu_answer  # Use LLM comparison directly
    elif domain == 'humaneval':
        async def check_humaneval(pred, truth):
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, humaneval_check_code, pred, truth)
        return check_humaneval
    elif domain in ('math', 'gsm8k'):
        return llm_judge_math_answer  # Already async
    else:
        async def check_default(pred, truth):
            return check_answer(pred, truth)
        return check_default


# Roles that provide auxiliary analysis rather than evaluable answers.
# These roles use alpha=0 (final reward only) since their node-level reward is meaningless.
NON_ANSWER_ROLES = frozenset({
    # HumanEval roles - these don't produce evaluable code
    'Project Manager',      # HumanEval: Design patterns and code structure
    'Algorithm Designer',   # HumanEval: Algorithm design and pseudocode
    'Test Analyst',         # HumanEval: Test cases and edge conditions
    # Note: MMLU roles removed - aggregators now output A/B/C/D answers and can be evaluated
})


class LearnableMultiNodeGraph(nn.Module):
    """
    Graph of LearnableMultiNodes with configurable topology and learnable edges.

    Each node in the graph is a LearnableMultiNode with its own LLMSelector.
    Nodes are connected according to a topology (Chain, FullConnected, etc.),
    and edges can be learned via policy gradient.

    Attributes:
        learnable_nodes: ModuleDict mapping role names to LearnableMultiNode instances
        topology: Dict mapping each role to its predecessor roles
        decision_node: Final aggregator that synthesizes all outputs
        mode: Topology mode string
        num_rounds: Number of iteration rounds
        alpha: Mixing coefficient for reward (node vs final)
    """

    def __init__(
        self,
        multi_node_configs: List[Dict[str, Any]],
        decision_method: str = 'FinalRefer',
        domain: str = 'gsm8k',
        mode: str = 'Chain',
        num_rounds: int = 1,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        initial_temperature: float = 2.0,
        edge_temperature: float = 1.0,
        use_llm_for_task_meta: bool = True,
        alpha: float = 0.5,
        optimized_spatial: bool = True,
        optimized_temporal: bool = False,
        decision_llm: str = 'gpt-4o-mini',
        initial_edge_probability: float = 0.5,
        cost_sensitivity: float = 500.0,
        share_selector: bool = False,
        inference_strategy: str = "standard",
        rotation_top_n: int = 2,
        use_edge_sqrt_transform: bool = False,
        use_qwen3_80b: bool = False,
        random_llm: bool = False,
        random_graph: bool = False,
        full_graph: bool = False,
        enable_skip_learning: bool = False,
        keep_top_k: int = 2,
        excluded_llm_indices: Optional[List[int]] = None,
        fixed_llm: bool = False,
        aggregator_only: bool = False,
    ):
        """
        Initialize the LearnableMultiNodeGraph.

        Args:
            multi_node_configs: List of config dicts, each containing:
                - role (str, required): Role name, e.g., "Mathematical Analyst"
                - num_workers (int, default=3): Number of workers
                - agent_name (str, default="MathSolver"): Agent type for workers
                - aggregator_method (str, default="FinalMultiAgentAggregator"): Aggregator type
            decision_method: Decision node type (e.g., 'FinalRefer')
            domain: Domain context (e.g., 'gsm8k')
            mode: Topology mode ('Chain', 'FullConnected', 'Star', 'Debate', etc.)
            num_rounds: Number of iteration rounds
            embed_dim: Embedding dimension for LLMSelector
            hidden_dim: Hidden dimension for LLMSelector
            initial_temperature: Initial Gumbel-Softmax temperature for LLM selection
            edge_temperature: Temperature for edge sampling
            use_llm_for_task_meta: Whether to use LLM for task metadata
            alpha: Reward mixing coefficient (0 = only final, 1 = only node)
            optimized_spatial: Whether to learn spatial edges
            optimized_temporal: Whether to learn temporal edges
            decision_llm: Fixed LLM for decision node
            initial_edge_probability: Initial probability for edges
            cost_sensitivity: Cost sensitivity for reward function
            share_selector: Whether to share a single LLMSelector across all nodes
            inference_strategy: LLM selection strategy ("standard" or "topN_rotation")
            rotation_top_n: Number of top LLMs to rotate for topN_rotation strategy
            use_edge_sqrt_transform: Whether to use sqrt transform for edge sampling
            use_qwen3_80b: Restrict LLM selection to OpenRouter models only (exclude GPT and DeepSeek-v3)
            enable_skip_learning: Enable learning to skip roles based on relative performance
            keep_top_k: Number of top-performing roles to keep (others can be skipped)
            excluded_llm_indices: List of LLM indices to exclude from selection (e.g., [2] for deepseek-r1-14b)
            fixed_llm: If True, use decision_llm for all workers/aggregators (no LLM selection)
            aggregator_only: If True, skip workers in all nodes (baseline mode - aggregator answers directly)
        """
        super().__init__()

        self.domain = domain
        self.mode = mode
        self.num_rounds = num_rounds
        self.alpha = alpha
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.edge_temperature = edge_temperature
        self.cost_sensitivity = cost_sensitivity
        self.share_selector = share_selector
        self.use_llm_for_task_meta = use_llm_for_task_meta
        self.inference_strategy = inference_strategy
        self.rotation_top_n = rotation_top_n
        self.use_edge_sqrt_transform = use_edge_sqrt_transform
        self.use_qwen3_80b = use_qwen3_80b
        self.random_llm = random_llm
        self.random_graph = random_graph
        self.full_graph = full_graph
        self.enable_skip_learning = enable_skip_learning
        self.keep_top_k = keep_top_k
        self.excluded_llm_indices = excluded_llm_indices
        self.fixed_llm = fixed_llm
        self.aggregator_only = aggregator_only
        self.id = shortuuid.ShortUUID().random(length=4)

        # Extract roles in order
        self.roles = [config['role'] for config in multi_node_configs]
        self.num_nodes = len(self.roles)

        # Create shared selector if requested
        if share_selector:
            self.shared_selector = LLMSelector(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_llms=len(LLM_OPTIONS),
                temperature=initial_temperature,
                random_llm=self.random_llm,
            )
        else:
            self.shared_selector = None

        # Create LearnableMultiNode for each role
        self.learnable_nodes = nn.ModuleDict()
        for config in multi_node_configs:
            role = config['role']
            node_id = f"mn_{role.replace(' ', '_')}_{self.id}"
            self.learnable_nodes[role] = LearnableMultiNode(
                num_workers=config.get('num_workers', 3),
                agent_name=config.get('agent_name', 'MathSolver'),
                aggregator_method=config.get('aggregator_method', 'FinalMultiAgentAggregator'),
                domain=domain,
                role=role,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                initial_temperature=initial_temperature,
                use_llm_for_task_meta=use_llm_for_task_meta,
                node_id=node_id,
                external_selector=self.shared_selector,
                use_qwen3_80b=use_qwen3_80b,
                random_llm=self.random_llm,
                allow_aggregator_skip=self.enable_skip_learning,
                excluded_llm_indices=excluded_llm_indices,
                fixed_llm=self.fixed_llm,
                fixed_llm_name=decision_llm if self.fixed_llm else None,
                aggregator_only=self.aggregator_only,
            )

        # Create decision node (fixed LLM)
        self.decision_node = AgentRegistry.get(
            decision_method,
            domain=domain,
            llm_name=decision_llm,
        )
        self.decision_llm = decision_llm

        # Build topology
        spatial_masks, temporal_masks = get_mode_masks(mode, self.num_nodes)
        self.topology = self._build_topology_from_masks(spatial_masks)

        # Learnable edge parameters
        num_potential_edges = self.num_nodes * self.num_nodes
        if num_potential_edges == 0:
            num_potential_edges = 1  # Avoid empty tensors

        # Spatial masks and logits
        self.spatial_masks = nn.Parameter(
            torch.tensor(spatial_masks, dtype=torch.float),
            requires_grad=False
        )
        init_logit = (
            torch.log(torch.tensor(initial_edge_probability / (1 - initial_edge_probability)))
            if optimized_spatial else 10.0
        )
        self.spatial_logits = nn.Parameter(
            torch.ones(num_potential_edges) * init_logit,
            requires_grad=optimized_spatial
        )

        # Temporal masks and logits
        self.temporal_masks = nn.Parameter(
            torch.tensor(temporal_masks, dtype=torch.float),
            requires_grad=False
        )
        init_temporal_logit = (
            torch.log(torch.tensor(initial_edge_probability / (1 - initial_edge_probability)))
            if optimized_temporal else 10.0
        )
        self.temporal_logits = nn.Parameter(
            torch.ones(num_potential_edges) * init_temporal_logit,
            requires_grad=optimized_temporal
        )

        # Task metadata cache for query enhancement
        self.task_cache = TaskMetaCache()

        # GCN-based edge learning (only if optimizing spatial)
        self.use_gcn_for_edges = optimized_spatial

        if self.use_gcn_for_edges:
            # Pre-compute static node features from role descriptions
            self.register_buffer(
                'node_features',
                self._construct_node_features(multi_node_configs)
            )

            # Prior adjacency matrix from topology mode (convert masks to edge_index)
            self.register_buffer(
                'role_adj_edge_index',
                self._masks_to_edge_index()
            )

            # SimpleGCN: input = node_feat(1024) + query_feat(1024) = 2048
            self.simple_gcn = SimpleGCN(input_dim=2048, output_dim=128)

            # Edge scorer MLP
            self.edge_scorer = EdgeScorer(node_emb_dim=128, hidden_dim=64)

    def _construct_node_features(self, multi_node_configs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Build node features from role descriptions via sentence embedding.

        Args:
            multi_node_configs: List of node configuration dicts

        Returns:
            Tensor of shape (N, 1024) with role embeddings
        """
        features = []
        for config in multi_node_configs:
            role = config['role']
            agent_name = config.get('agent_name', 'Agent')
            description = f"{role}: {agent_name}"
            emb = get_sentence_embedding(description)
            features.append(emb)
        return torch.tensor(np.array(features), dtype=torch.float32)

    def _masks_to_edge_index(self) -> torch.Tensor:
        """
        Convert spatial_masks to edge_index format for GCN.

        Returns:
            Edge index tensor of shape (2, E) in COO format
        """
        N = self.num_nodes
        adj = self.spatial_masks.view(N, N)
        edge_index, _ = dense_to_sparse(adj)
        return edge_index

    def _build_query_string(self, task: str) -> str:
        """
        Build enhanced query string with task metadata.

        Combines the original task description with extracted metadata
        (type, difficulty, domain, key_skills) for richer query embedding.

        Args:
            task: Original task description

        Returns:
            Enhanced query string for embedding
        """
        
        meta = self.task_cache.get_or_extract(task, use_llm=self.use_llm_for_task_meta)
        query_parts = [
            f"Task: {task}",
            f"Type: {meta.get('type', 'general')}",
            f"Difficulty: {meta.get('difficulty', 'medium')}",
            f"Domain: {meta.get('domain', 'general')}",
            f"Key Skills: {', '.join(meta.get('key_skills', []))}",
        ]
        return "\n".join(query_parts)

    def compute_spatial_logits(self, task: str, apply_clamp: bool = True) -> torch.Tensor:
        """
        Compute edge logits using SimpleGCN on node + query features.

        Args:
            task: Task description string
            apply_clamp: If True, apply soft_clamp to logits. Default True for training.
                        Set to False to get raw logits for inspection.

        Returns:
            Edge logits tensor of shape (N*N,)
        """
        if not self.use_gcn_for_edges:
            logits = self.spatial_logits
        else:
            device = self.node_features.device

            # Build enhanced query string with metadata
            query_string = self._build_query_string(task)

            # Embed query (1024-dim)
            query_emb = torch.tensor(
                get_sentence_embedding(query_string),
                dtype=torch.float32,
                device=device
            )

            # Expand query to all nodes: (N, 1024)
            query_expanded = query_emb.unsqueeze(0).expand(self.num_nodes, -1)

            # Concatenate node features with query features: (N, 2048)
            combined = torch.cat([self.node_features, query_expanded], dim=1)

            # SimpleGCN forward pass: (N, 128)
            node_emb = self.simple_gcn(combined, self.role_adj_edge_index)

            # Compute all pairwise edge logits: (N*N,)
            logits = self.edge_scorer.forward_all(node_emb, apply_clamp=apply_clamp)

        # Handle ablation modes
        if self.random_graph:
            print(f"[SpatialLogits] random_graph=True, zeroing out logits. Original logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            zeroed_logits = torch.zeros_like(logits)
            print(f"[SpatialLogits] After zeroing: min={zeroed_logits.min().item():.4f}, max={zeroed_logits.max().item():.4f}, mean={zeroed_logits.mean().item():.4f}")
            return zeroed_logits
        elif self.full_graph:
            print(f"[SpatialLogits] full_graph=True, setting logits to 100000. Original logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            full_logits = torch.full_like(logits, 100000.0)
            print(f"[SpatialLogits] After setting to 100000: min={full_logits.min().item():.4f}, max={full_logits.max().item():.4f}, mean={full_logits.mean().item():.4f}")
            return full_logits

        return logits

    def _build_topology_from_masks(self, spatial_masks: List[int]) -> Dict[str, List[str]]:
        """
        Build topology dictionary from spatial masks.

        Args:
            spatial_masks: Flattened N×N mask array

        Returns:
            Dict mapping each role to its predecessor roles
        """
        N = self.num_nodes
        topology = {role: [] for role in self.roles}

        for j in range(N):  # destination
            for i in range(N):  # source
                idx = j * N + i
                if spatial_masks[idx] == 1 and i != j:
                    # i -> j connection exists
                    topology[self.roles[j]].append(self.roles[i])

        return topology

    def _compute_execution_order(self) -> List[str]:
        """
        Compute topological execution order using Kahn's algorithm.

        Returns:
            List of role names in execution order
        """
        # Compute in-degrees
        in_degree = {role: len(preds) for role, preds in self.topology.items()}

        # Start with nodes that have no predecessors
        queue = deque([role for role, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            role = queue.popleft()
            order.append(role)

            # Reduce in-degree for successors
            for other_role in self.roles:
                if role in self.topology[other_role]:
                    in_degree[other_role] -= 1
                    if in_degree[other_role] == 0:
                        queue.append(other_role)

        # Handle cycles or disconnected nodes
        for role in self.roles:
            if role not in order:
                order.append(role)

        return order

    def _compute_edge_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of Bernoulli distribution for edge probabilities.

        Entropy H(p) = -p*log(p) - (1-p)*log(1-p)

        Args:
            probs: Tensor of edge probabilities

        Returns:
            Total entropy (scalar)
        """
        eps = 1e-10
        probs = torch.clamp(probs, min=eps, max=1.0 - eps)
        return (-probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)).sum()

    def _sample_edges(
        self,
        round_idx: int,
        task: str = "",
        is_inference: bool = False,
        override_spatial_adj: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], torch.Tensor, torch.Tensor]:
        """
        Sample active edges from logits with masks.

        Args:
            round_idx: Current round index (for temporal edges)
            task: Task description for query-conditioned logits
            is_inference: Whether to use inference-time edge sampling (sqrt transform)
            override_spatial_adj: Optional flattened adjacency tensor (N*N,) to override
                                  self.spatial_masks. Used for random graph pool in Stage 1.

        Returns:
            active_spatial: Dict[role, List[predecessor_roles]]
            active_temporal: Dict[role, List[predecessor_roles]]
            edge_log_probs: Tensor of log probabilities for sampled edges
            edge_entropy: Entropy of edge sampling distribution
        """
        N = self.num_nodes
        device = next(self.parameters()).device
        log_probs = []
        all_edge_probs = []  # Collect probabilities for entropy computation

        # Use override spatial adjacency if provided, otherwise use learned masks
        spatial_mask = override_spatial_adj if override_spatial_adj is not None else self.spatial_masks

        # Compute spatial logits from GCN (query-conditioned)
        # Note: EdgeScorer.forward_all already applies soft_clamp, so no need to clamp here
        spatial_logits = self.compute_spatial_logits(task)

        # Sample spatial edges
        active_spatial = {role: [] for role in self.roles}
        for j in range(N):
            for i in range(N):
                idx = j * N + i
                if spatial_mask[idx] == 1 and i != j:
                    logit = spatial_logits[idx]

                    # Apply sqrt transform in inference mode if enabled
                    if is_inference and self.use_edge_sqrt_transform:
                        # transformed_logit = sign(logit) * sqrt(abs(logit))
                        transformed_logit = torch.sign(logit) * torch.sqrt(torch.abs(logit) + 1e-8)
                        prob = torch.sigmoid(transformed_logit)
                        all_edge_probs.append(prob)
                        # Always use probabilistic sampling with sqrt transform
                        sample = torch.bernoulli(prob)
                        if sample == 1:
                            active_spatial[self.roles[j]].append(self.roles[i])
                            log_probs.append(torch.log(prob + 1e-10))
                        else:
                            log_probs.append(torch.log(1 - prob + 1e-10))
                    else:
                        # Original behavior
                        prob = torch.sigmoid(logit / self.edge_temperature)
                        all_edge_probs.append(prob)

                        if self.optimized_spatial and self.training:
                            # Sample during training
                            sample = torch.bernoulli(prob)
                            if sample == 1:
                                active_spatial[self.roles[j]].append(self.roles[i])
                                log_probs.append(torch.log(prob + 1e-10))
                            else:
                                log_probs.append(torch.log(1 - prob + 1e-10))
                        else:
                            # Use deterministic threshold during eval
                            if prob > 0.5:
                                active_spatial[self.roles[j]].append(self.roles[i])

        # Sample temporal edges (only if round > 0)
        active_temporal = {role: [] for role in self.roles}
        if round_idx > 0:
            for j in range(N):
                for i in range(N):
                    idx = j * N + i
                    if self.temporal_masks[idx] == 1:
                        logit = self.temporal_logits[idx]
                        # Use soft clamp for temporal logits to maintain gradient flow
                        # Default limit=5.0 keeps logits where sigmoid has meaningful gradient
                        logit = soft_clamp(logit.unsqueeze(0)).squeeze(0)

                        # Apply sqrt transform in inference mode if enabled
                        if is_inference and self.use_edge_sqrt_transform:
                            transformed_logit = torch.sign(logit) * torch.sqrt(torch.abs(logit) + 1e-8)
                            prob = torch.sigmoid(transformed_logit)
                            all_edge_probs.append(prob)
                            # Always use probabilistic sampling with sqrt transform
                            sample = torch.bernoulli(prob)
                            if sample == 1:
                                active_temporal[self.roles[j]].append(self.roles[i])
                                log_probs.append(torch.log(prob + 1e-10))
                            else:
                                log_probs.append(torch.log(1 - prob + 1e-10))
                        else:
                            prob = torch.sigmoid(logit / self.edge_temperature)
                            all_edge_probs.append(prob)

                            if self.optimized_temporal and self.training:
                                sample = torch.bernoulli(prob)
                                if sample == 1:
                                    active_temporal[self.roles[j]].append(self.roles[i])
                                    log_probs.append(torch.log(prob + 1e-10))
                                else:
                                    log_probs.append(torch.log(1 - prob + 1e-10))
                            else:
                                if prob > 0.5:
                                    active_temporal[self.roles[j]].append(self.roles[i])

        # Stack log probs
        if log_probs:
            edge_log_probs = torch.stack(log_probs).sum()
        else:
            # Create zero that maintains gradient connection to spatial_logits
            # This ensures gradients can flow back even when no edges are sampled
            edge_log_probs = spatial_logits.sum() * 0.0

        # Compute edge entropy for regularization
        if all_edge_probs:
            edge_probs_tensor = torch.stack(all_edge_probs)
            edge_entropy = self._compute_edge_entropy(edge_probs_tensor)
        else:
            edge_entropy = torch.tensor(0.0, device=device, requires_grad=True)

        return active_spatial, active_temporal, edge_log_probs, edge_entropy

    def compute_skip_targets(
        self,
        node_results: Dict[str, Dict],
        keep_top_k: int = 2
    ) -> Dict[str, float]:
        """
        Compute skip target (Hard Label based on Top-K) for each answer role.

        Only considers answer roles (alpha > 0). NON_ANSWER_ROLES don't participate
        because they don't output evaluable solutions.

        Args:
            node_results: Dict mapping role to node execution results
            keep_top_k: Number of top-performing roles to keep (others get skip target 1.0)

        Returns:
            Dict mapping role to skip target:
            - 0.0: Don't skip (role is in Top-K)
            - 1.0: Skip (role is not in Top-K)
        """
        # Only consider answer roles (alpha > 0)
        # NON_ANSWER_ROLES don't output solutions and can't be evaluated
        role_rewards = []
        for role, result in node_results.items():
            if role in NON_ANSWER_ROLES:
                continue  # Skip non-answer roles

            # Get reward (use -1.0 for already skipped nodes)
            if result.get('is_skipped', False):
                reward = -1.0
            else:
                reward = result.get('reward', 0.0)

            role_rewards.append((role, reward))

        # If no answer roles, return empty
        if not role_rewards:
            return {}

        # Sort by reward descending
        role_rewards.sort(key=lambda x: x[1], reverse=True)

        # Top-K get 0 (keep), others get 1 (skip)
        skip_targets = {}
        for i, (role, reward) in enumerate(role_rewards):
            if i < keep_top_k:
                skip_targets[role] = 0.0  # Keep (in Top-K)
            else:
                skip_targets[role] = 1.0  # Skip (not in Top-K)

        return skip_targets

    def compute_role_skip_probs(
        self,
        input_dict: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute skip probabilities for each role WITHOUT executing LLM calls.

        Used at inference time to determine which roles to skip before execution.

        Args:
            input_dict: Dict containing 'task' key

        Returns:
            Dict mapping role -> {
                'aggregator_skip_prob': float,  # Skip probability (0-1)
                'all_probs': List[float],       # All action probabilities
                'is_skip_argmax': bool,         # Whether skip is the argmax action
            }
        """
        task = input_dict.get('task', '')
        device = next(self.parameters()).device

        # Use first node's embedder (all nodes share the same embedder class)
        first_node = next(iter(self.learnable_nodes.values()))
        embedder = first_node.embedder

        # Get LLM embeddings
        llm_emb = embedder.get_llm_embeddings(device=device)

        role_skip_info = {}

        for role, node in self.learnable_nodes.items():
            # Get role-aware task embedding
            role_desc = node.role_description if hasattr(node, 'role_description') else ''
            task_role_emb = embedder.get_task_role_embedding(
                task, role=role, role_description=role_desc, device=device
            )

            # Get metadata embedding if available
            meta_emb = None
            if hasattr(self, 'task_cache') and self.task_cache is not None:
                try:
                    metadata = self.task_cache.get_metadata(task)
                    meta_emb = embedder.get_meta_embedding(metadata, device=device)
                except Exception:
                    pass

            # Get selector (shared or per-node)
            selector = self.shared_selector if self.share_selector else node.selector

            # Compute selection probabilities
            # Note: allow_aggregator_skip=True is critical here to get actual skip probabilities
            # Without it, aggregator's skip logit would be masked to -1e10, giving ~0 probability
            num_positions = node.num_workers + 1  # workers + aggregator
            probs = selector.get_selection_probs(
                num_positions=num_positions,
                llm_emb=llm_emb,
                task_emb=task_role_emb,
                meta_emb=meta_emb,
                aggregator_mask=True,
                is_training=False,
                allow_aggregator_skip=True,
            )

            # Aggregator is the last position, skip is the last action (index -1)
            aggregator_probs = probs[-1]  # Shape: (num_llms + 1,) including skip
            aggregator_skip_prob = aggregator_probs[-1].item()

            # Check if skip is argmax
            is_skip_argmax = (aggregator_probs.argmax().item() == len(aggregator_probs) - 1)

            role_skip_info[role] = {
                'aggregator_skip_prob': aggregator_skip_prob,
                'all_probs': aggregator_probs.tolist(),
                'is_skip_argmax': is_skip_argmax,
            }

        return role_skip_info

    def determine_roles_to_skip(
        self,
        role_skip_info: Dict[str, Dict[str, Any]],
        topK_roles_removed: int = 3,
    ) -> Set[str]:
        """
        Skip at most K roles where skip is argmax action.

        - If >= K roles have skip as argmax: skip top-K by skip_prob
        - If < K roles have skip as argmax: skip only those (could be 0)
        - Always keep at least 1 role

        Args:
            role_skip_info: Output from compute_role_skip_probs()
            topK_roles_removed: Maximum number of roles to skip (default=3)

        Returns:
            Set of role names to skip
        """
        # Filter to only consider answer roles (not in NON_ANSWER_ROLES)
        answer_roles_info = {
            role: info for role, info in role_skip_info.items()
            if role not in NON_ANSWER_ROLES
        }

        # Protect: at least 1 role must remain
        max_skippable = len(answer_roles_info) - 1
        k = min(topK_roles_removed, max(0, max_skippable))

        if k <= 0:
            return set()

        # Find roles where skip is argmax
        argmax_skip_roles = [
            (role, info['aggregator_skip_prob'])
            for role, info in answer_roles_info.items()
            if info['is_skip_argmax']
        ]

        if len(argmax_skip_roles) >= k:
            # Skip top-K by skip_prob
            argmax_skip_roles.sort(key=lambda x: x[1], reverse=True)
            return {role for role, _ in argmax_skip_roles[:k]}
        else:
            # Skip all argmax-skip roles (could be 0)
            return {role for role, _ in argmax_skip_roles}

    async def forward_with_role_skip(
        self,
        input_dict: Dict[str, Any],
        roles_to_skip: Set[str],
        ground_truth: Optional[str] = None,
        hard: bool = True,
        is_training: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute forward pass with specified roles pre-skipped.

        Skipped roles:
        - Do NOT execute LLM calls (saves cost)
        - Do NOT participate in message passing
        - Are marked with is_skipped=True in results

        Args:
            input_dict: Dict with 'task' key
            roles_to_skip: Set of role names to skip
            ground_truth: For reward computation
            hard: Use hard selection (default True for inference)
            is_training: Training mode flag (default False for inference)

        Returns:
            Same format as forward(), with skipped roles marked
        """
        # Execute forward with force_skip_roles parameter (thread-safe, no state mutation)
        result = await self.forward(
            input_dict,
            ground_truth=ground_truth,
            hard=hard,
            is_training=is_training,
            force_skip_roles=set(roles_to_skip),
        )

        # Mark forced skips in result
        result['forced_skip_roles'] = list(roles_to_skip)

        return result

    async def forward(
        self,
        input: Dict[str, Any],
        ground_truth: str,
        hard: bool = False,
        is_training: bool = True,
        force_skip_roles: Optional[Set[str]] = None,
        override_spatial_adj: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Execute the entire graph and compute rewards.

        Args:
            input: Input dict with 'task' key
            ground_truth: Ground truth answer for reward computation
            hard: Whether to use hard selection (for evaluation)
            is_training: If True, clamp logits; if False, no clamp and use T=10
            force_skip_roles: Set of role names to force skip (for parallel-safe role skip)
            override_spatial_adj: Optional flattened adjacency tensor (N*N,) to override
                                  spatial masks. Used for random graph pool in Stage 1.

        Returns:
            Dictionary containing:
            - 'final_output': Decision node's output
            - 'final_prediction': Extracted prediction from final output
            - 'final_reward': Reward based on final output
            - 'final_solved': Whether final output is correct
            - 'node_results': Dict mapping role to node results
            - 'total_log_probs': Sum of all log_probs
            - 'total_cost': Total API cost
            - 'effective_rewards': Dict of computed effective rewards per node
            - 'sampled_edges': Dict with 'spatial' and 'temporal' edge info per round
        """
        # Get domain-aware answer extraction and checking functions
        answer_extractor = get_answer_extractor(self.domain)
        answer_checker = get_answer_checker(self.domain)

        node_results = {}
        node_outputs = {}  # For building spatial_info
        previous_outputs = {}  # For temporal info
        total_cost = 0.0
        total_log_probs = None  # Use None to maintain gradient flow
        total_edge_log_probs = None  # Track edge log_probs separately for loss
        total_entropy = None  # Track total entropy for regularization
        skipped_roles = set()  # Track roles that were skipped

        # Track sampled edges across rounds
        sampled_edges = {'spatial': {}, 'temporal': {}}

        # Execution order
        execution_order = self._compute_execution_order()

        # Extract task for query-conditioned edge computation
        task = input.get('task', '')

        # Determine if using inference strategy (for sqrt transform and topN rotation)
        is_inference = (not is_training) and (self.inference_strategy == "topN_rotation")

        # Compute spatial logits once for Hoyer regularization
        # Store before the round loop to avoid recomputation
        all_spatial_logits = None
        if self.optimized_spatial and self.use_gcn_for_edges:
            all_spatial_logits = self.compute_spatial_logits(task)

        for round_idx in range(self.num_rounds):
            # Sample edges for this round (pass task for GCN-based logits)
            active_spatial, active_temporal, edge_log_probs, edge_entropy = self._sample_edges(
                round_idx, task=task, is_inference=is_inference,
                override_spatial_adj=override_spatial_adj
            )
            # Accumulate edge log probs with gradient-preserving addition
            if total_edge_log_probs is None:
                total_edge_log_probs = edge_log_probs
            else:
                total_edge_log_probs = total_edge_log_probs + edge_log_probs
            total_log_probs = total_edge_log_probs  # Keep in sync

            # Accumulate edge entropy
            if total_entropy is None:
                total_entropy = edge_entropy
            else:
                total_entropy = total_entropy + edge_entropy

            # Store sampled edges for this round
            sampled_edges['spatial'][round_idx] = active_spatial
            sampled_edges['temporal'][round_idx] = active_temporal

            # Clear node outputs for this round
            current_round_outputs = {}

            # Execute nodes in topological order
            for role in execution_order:
                node = self.learnable_nodes[role]

                # Build spatial_info from predecessors in this round
                spatial_info = {}
                for pred_role in active_spatial.get(role, []):
                    if pred_role in current_round_outputs:
                        spatial_info[self.learnable_nodes[pred_role].node_id if hasattr(self, 'learnable_nodes') else pred_role] = {
                            'role': pred_role,
                            'output': current_round_outputs[pred_role],
                        }

                # Build temporal_info from previous round
                temporal_info = {}
                if round_idx > 0:
                    for pred_role in active_temporal.get(role, []):
                        if pred_role in previous_outputs:
                            temporal_info[self.learnable_nodes[pred_role].node_id] = {
                                'role': pred_role,
                                'output': previous_outputs[pred_role],
                            }

                # Check if this role should be force-skipped
                should_force_skip = force_skip_roles is not None and role in force_skip_roles

                # Execute node (returns dict with output, log_probs, cost, etc.)
                node_result = await node.forward(
                    input,
                    spatial_info=spatial_info if spatial_info else None,
                    temporal_info=temporal_info if temporal_info else None,
                    hard=hard,
                    is_training=is_training,
                    inference_strategy=self.inference_strategy,
                    rotation_top_n=self.rotation_top_n,
                    force_skip=should_force_skip,
                )

                # Extract values from result dict
                output = node_result['output']
                log_probs = node_result['log_probs']
                cost = node_result['cost']
                llm_selection = node_result['llm_selection']
                node_entropy = node_result['entropy']
                is_skipped = node_result.get('is_skipped', False)

                # Track skipped roles
                if is_skipped:
                    skipped_roles.add(role)
                    # Skipped nodes don't contribute output to spatial info
                else:
                    current_round_outputs[role] = output

                total_cost += cost
                # Accumulate node log probs with gradient-preserving addition
                node_log_sum = log_probs.sum()
                if total_log_probs is None:
                    total_log_probs = node_log_sum
                else:
                    total_log_probs = total_log_probs + node_log_sum

                # Accumulate node entropy
                if total_entropy is None:
                    total_entropy = node_entropy
                else:
                    total_entropy = total_entropy + node_entropy

                # Compute per-node reward
                if is_skipped:
                    prediction = ''
                    solved = False
                    reward = 0.0
                else:
                    prediction = answer_extractor(output)
                    solved = await answer_checker(prediction, ground_truth)
                    reward = reward_v2(solved, cost, self.cost_sensitivity)

                node_results[role] = {
                    'output': output,
                    'prediction': prediction,
                    'reward': reward,
                    'solved': solved,
                    'cost': cost,
                    'llm_selection': llm_selection,
                    'log_probs': log_probs,
                    'prompt_tokens': node_result.get('prompt_tokens', 0),
                    'completion_tokens': node_result.get('completion_tokens', 0),
                    'is_skipped': is_skipped,
                    'skip_probs': node_result.get('skip_probs'),
                    'aggregator_skip_prob': node_result.get('aggregator_skip_prob'),
                }

            # Update previous outputs for next round
            previous_outputs = current_round_outputs.copy()
            node_outputs = current_round_outputs.copy()

        # Execute decision node
        # Build spatial info for decision node from non-skipped node outputs only
        decision_spatial_info = {}
        for role, output in node_outputs.items():
            if role not in skipped_roles:  # Only include non-skipped nodes
                decision_spatial_info[self.learnable_nodes[role].node_id] = {
                    'role': role,
                    'output': output,
                }

        # Prepare decision node input
        decision_input = {**input}
        decision_task = input.get('task', '')

        # Add non-skipped outputs to decision task
        outputs_text = []
        for role, output in node_outputs.items():
            if role not in skipped_roles:  # Only include non-skipped nodes
                outputs_text.append(f"Agent {role} output:\n{output}")

        enhanced_decision_task = (
            decision_task +
            "\n\nBased on the following agent outputs, provide the final answer:\n\n" +
            "\n\n".join(outputs_text) +
            "\n\nIMPORTANT: End your response with 'The answer is [your_answer]'"
        )
        decision_input['task'] = enhanced_decision_task

        # Execute decision node
        cost_before = Cost.instance().value
        final_output = await self.decision_node.async_execute(decision_input)
        if isinstance(final_output, list):
            final_output = final_output[0] if final_output else ""
        cost_after = Cost.instance().value
        decision_cost = cost_after - cost_before
        total_cost += decision_cost

        # Consensus override: if all non-skipped worker nodes agree, use their answer directly
        node_predictions = [result['prediction'] for role, result in node_results.items()
                           if not result.get('is_skipped', False) and result['prediction']]
        if len(set(node_predictions)) == 1 and node_predictions:
            # All nodes agree - use consensus directly (avoids extraction errors)
            final_prediction = node_predictions[0]
        else:
            # Nodes disagree - use decision node's extracted answer
            final_prediction = answer_extractor(final_output)
        final_solved = await answer_checker(final_prediction, ground_truth)
        final_reward = reward_v2(final_solved, decision_cost, self.cost_sensitivity)

        # Compute effective rewards for each node (role-aware alpha)
        effective_rewards = {}
        for role, result in node_results.items():
            role_alpha = 0.0 if role in NON_ANSWER_ROLES else self.alpha
            effective_rewards[role] = compute_mixed_reward(
                result['reward'], final_reward, role_alpha
            )

        # Ensure we return valid tensors even if no log probs were accumulated
        device = next(self.parameters()).device
        if total_log_probs is None:
            total_log_probs = torch.tensor(0.0, device=device, requires_grad=True)
        if total_edge_log_probs is None:
            total_edge_log_probs = torch.tensor(0.0, device=device, requires_grad=True)
        if total_entropy is None:
            total_entropy = torch.tensor(0.0, device=device, requires_grad=True)

        # Compute skip targets for skip loss
        skip_targets = self.compute_skip_targets(node_results, self.keep_top_k)

        return {
            'final_output': final_output,
            'final_prediction': final_prediction,
            'final_reward': final_reward,
            'final_solved': final_solved,
            'node_results': node_results,
            'total_log_probs': total_log_probs,
            'total_cost': total_cost,
            'effective_rewards': effective_rewards,
            'sampled_edges': sampled_edges,
            'edge_log_probs': total_edge_log_probs,  # For edge loss computation
            'total_entropy': total_entropy,  # For entropy regularization
            'spatial_logits': all_spatial_logits,  # For Hoyer sparsity regularization
            'skipped_roles': list(skipped_roles),  # For skip statistics
            'skip_targets': skip_targets,  # For skip loss computation
        }

    def compute_loss(
        self,
        forward_result: Dict[str, Any],
        entropy_coef: float = 0.8,
        hoyer_coef: float = 0.8,
        skip_coef: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss with entropy, Hoyer sparsity, and skip regularization.

        For each node i:
            effective_reward_i = alpha * node_reward_i + (1-alpha) * final_reward
            loss_i = -log_probs_i.sum() * effective_reward_i

        Total loss = sum(loss_i) + edge_loss - entropy_coef * total_entropy
                     + hoyer_coef * hoyer_penalty + skip_coef * skip_loss

        Args:
            forward_result: Output from forward() method
            entropy_coef: Coefficient for entropy regularization (default: 0.8)
                         Higher values encourage more exploration.
            hoyer_coef: Coefficient for Hoyer sparsity regularization (default: 0.8)
                       Higher values encourage sparser edge configurations.
            skip_coef: Coefficient for skip loss (default: 0.0)
                      Only applies when enable_skip_learning is True.

        Returns:
            Scalar loss tensor
        """
        total_loss = None  # Use None to maintain gradient flow
        node_results = forward_result['node_results']
        effective_rewards = forward_result['effective_rewards']

        for role, result in node_results.items():
            log_probs = result['log_probs']
            eff_reward = effective_rewards[role]
            node_loss = -log_probs.sum() * eff_reward
            if total_loss is None:
                total_loss = node_loss
            else:
                total_loss = total_loss + node_loss

        # Edge loss: global structure decisions use final_reward
        # Edges define collaboration structure and affect all nodes collectively
        if self.optimized_spatial or self.optimized_temporal:
            edge_log_probs = forward_result.get('edge_log_probs')
            if edge_log_probs is not None:
                final_reward = forward_result['final_reward']
                edge_loss = -edge_log_probs * final_reward
                if total_loss is None:
                    total_loss = edge_loss
                else:
                    total_loss = total_loss + edge_loss

        # Return zero loss if nothing to optimize (shouldn't happen normally)
        if total_loss is None:
            total_loss = torch.tensor(0.0, requires_grad=True)

        # Entropy regularization: maximize entropy = minimize -entropy
        # This prevents policy collapse where the model becomes too deterministic
        if entropy_coef > 0:
            total_entropy = forward_result.get('total_entropy')
            if total_entropy is not None:
                total_loss = total_loss - entropy_coef * total_entropy

        # Hoyer sparsity regularization: encourage sparse edge distribution
        # Hoyer = L1 / sqrt(L2^2) approximates L0 norm
        # Minimizing this pushes distribution toward sparse (few edges ~1, most ~0)
        if hoyer_coef > 0 and self.optimized_spatial:
            spatial_logits = forward_result.get('spatial_logits')
            if spatial_logits is not None:
                # Convert logits to probabilities
                spatial_probs = torch.sigmoid(spatial_logits / self.edge_temperature)

                # Apply topology mask (only consider allowed edges)
                masked_probs = spatial_probs * self.spatial_masks

                # Compute Hoyer regularization: L1 / sqrt(L2^2)
                l1_norm = masked_probs.sum()
                l2_squared = (masked_probs ** 2).sum()

                # Add epsilon for numerical stability
                epsilon = 1e-8
                hoyer_loss = l1_norm / torch.sqrt(l2_squared + epsilon)

                total_loss = total_loss + hoyer_coef * hoyer_loss

        # Skip loss: BCE with hard labels for answer roles only
        # This teaches the model which roles to skip based on relative performance
        if skip_coef > 0 and self.enable_skip_learning:
            skip_targets = forward_result.get('skip_targets', {})
            device = next(self.parameters()).device

            skip_loss = torch.tensor(0.0, device=device)
            num_targets = 0

            for role, target in skip_targets.items():
                result = node_results.get(role)
                if result is None:
                    continue

                skip_prob = result.get('aggregator_skip_prob')
                if skip_prob is None:
                    continue

                # BCE: -y * log(p) - (1-y) * log(1-p)
                target_tensor = torch.tensor(target, device=device)
                skip_prob_clamped = torch.clamp(skip_prob, 1e-7, 1 - 1e-7)

                bce = -target_tensor * torch.log(skip_prob_clamped) \
                      - (1 - target_tensor) * torch.log(1 - skip_prob_clamped)
                skip_loss = skip_loss + bce
                num_targets += 1

            # Average skip loss
            if num_targets > 0:
                skip_loss = skip_loss / num_targets
                if total_loss is None:
                    total_loss = skip_coef * skip_loss
                else:
                    total_loss = total_loss + skip_coef * skip_loss

        return total_loss

    def get_node(self, role: str) -> LearnableMultiNode:
        """Get a LearnableMultiNode by role name."""
        return self.learnable_nodes[role]

    def set_temperature(self, temperature: float):
        """Set Gumbel-Softmax temperature for all nodes.

        When using a shared selector, sets temperature on the shared selector.
        Otherwise, sets temperature on each node's individual selector.
        """
        if self.share_selector and self.shared_selector is not None:
            self.shared_selector.set_temperature(temperature)
        else:
            for node in self.learnable_nodes.values():
                node.set_temperature(temperature)

    def set_edge_temperature(self, temperature: float):
        """Set edge sampling temperature."""
        self.edge_temperature = temperature

    @property
    def temperature(self) -> float:
        """Get current temperature (from first node)."""
        first_node = next(iter(self.learnable_nodes.values()))
        return first_node.temperature

    def get_topology_summary(self) -> str:
        """Return a string summarizing the current topology."""
        lines = [f"Topology Mode: {self.mode}", "Connections:"]
        for role, preds in self.topology.items():
            if preds:
                lines.append(f"  {role} <- {preds}")
            else:
                lines.append(f"  {role} (no predecessors)")
        return "\n".join(lines)

    def get_config_summary(self) -> str:
        """Return a configuration summary for logging."""
        lines = [
            "=" * 80,
            "LearnableMultiNodeGraph Configuration",
            "=" * 80,
            f"Mode: {self.mode}",
            f"Num Nodes: {self.num_nodes}",
            f"Num Rounds: {self.num_rounds}",
            f"Alpha: {self.alpha}",
            f"Optimized Spatial: {self.optimized_spatial}",
            f"Optimized Temporal: {self.optimized_temporal}",
            f"Decision LLM: {self.decision_llm}",
            "",
            "Nodes:",
        ]
        for role, node in self.learnable_nodes.items():
            lines.append(f"  {role}: {node.num_workers} workers, agent={node.agent_name}")
        lines.extend(["", self.get_topology_summary()])
        return "\n".join(lines)

    def update_masks(self, threshold: float = 0.1):
        """
        Prune low-probability edges by setting mask to 0.

        Args:
            threshold: Edges with prob < threshold will be pruned
        """
        with torch.no_grad():
            spatial_probs = torch.sigmoid(self.spatial_logits / self.edge_temperature)
            low_prob_mask = spatial_probs < threshold
            self.spatial_masks.data[low_prob_mask] = 0.0

            if self.optimized_temporal:
                temporal_probs = torch.sigmoid(self.temporal_logits / self.edge_temperature)
                low_temp_mask = temporal_probs < threshold
                self.temporal_masks.data[low_temp_mask] = 0.0
