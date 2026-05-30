"""
MultiNodeGraph: A framework for managing and executing a network of MultiNodes.

This module provides the MultiNodeGraph class which enables multiple MultiNode instances
to work together, where each MultiNode contains workers with the same role and an aggregator.
MultiNode connectivity is controlled by the `mode` parameter and masks, while ROLE_CONNECTION
is only used for GCN adjacency matrix (prior knowledge for learning).
"""

import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio

from HieraMAS.graph.node import Node
from HieraMAS.graph.multi_node import MultiNode
from HieraMAS.agents.agent_registry import AgentRegistry
from HieraMAS.prompt.prompt_set_registry import PromptSetRegistry
from HieraMAS.llm.profile_embedding import get_sentence_embedding
from HieraMAS.gnn.gcn import GCN, MLP
from torch_geometric.utils import dense_to_sparse


def get_mode_masks(mode: str, num_multi_nodes: int) -> Tuple[List[int], List[int]]:
    """
    Generate spatial and temporal masks based on topology mode.

    Args:
        mode: Topology mode ('FullConnected', 'Chain', 'Debate', etc.)
        num_multi_nodes: Number of MultiNodes in the graph

    Returns:
        Tuple of (spatial_masks, temporal_masks) as flattened lists for N×N potential edges.
    """
    import random
    N = num_multi_nodes

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[0 for _ in range(N)] for _ in range(N)]

    elif mode == 'FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    elif mode == 'Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]

    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]

    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    elif mode == 'Layered':
        layer_num = 2
        layer_size = N // layer_num if layer_num > 0 else N
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        for layer in range(layer_num - 1):
            for i in range(layer * layer_size, (layer + 1) * layer_size):
                for j in range((layer + 1) * layer_size, min((layer + 2) * layer_size, N)):
                    if i < N and j < N:
                        fixed_spatial_masks[i][j] = 1
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    elif mode == 'Star':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        for i in range(1, N):
            fixed_spatial_masks[0][i] = 1  # Hub to spokes
            fixed_spatial_masks[i][0] = 1  # Spokes to hub
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    elif mode == 'Mesh':
        # Upper triangular connectivity
        fixed_spatial_masks = [[1 if i < j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Flatten to 1D lists
    spatial_flat = [fixed_spatial_masks[j][i] for j in range(N) for i in range(N)]
    temporal_flat = [fixed_temporal_masks[j][i] for j in range(N) for i in range(N)]

    return spatial_flat, temporal_flat


class MultiNodeGraph(ABC):
    """
    A framework for managing and executing a network of MultiNodes.

    Each MultiNode contains multiple worker nodes with the same role and an aggregator.
    MultiNode connectivity is controlled by the `mode` parameter (e.g., 'FullConnected',
    'Chain', 'Debate', etc.) and masks, enabling flexible topology configurations.
    ROLE_CONNECTION is only used for GCN adjacency matrix (prior knowledge for learning).

    Architecture:
        MultiNode A (role: "Mathematical Analyst")
        ├── Worker A1, A2, A3 (same role)
        └── Aggregator A
                ↓ (spatial connection controlled by mode + masks)
        MultiNode B (role: "Math Solver")
        ├── Worker B1, B2, B3 (same role)
        └── Aggregator B
            └── spatial_predecessors: [B1, B2, B3, Aggregator A]

    Attributes:
        domain (str): The domain for which this graph is used (e.g., "gsm8k")
        llm_name (str): The name of the LLM used for processing within nodes
        mode (str): Topology mode controlling MultiNode connections
        multi_nodes (Dict[str, MultiNode]): Collection of MultiNodes by ID
        decision_node (Node): Final aggregator that synthesizes all outputs
        potential_spatial_edges: All possible connections between MultiNode aggregators (N×N)
        potential_temporal_edges: All possible temporal connections (N×N)

    Methods:
        run(inputs, num_rounds): Execute the graph synchronously
        arun(inputs, num_rounds): Execute the graph asynchronously
    """

    def __init__(
        self,
        domain: str,
        llm_name: Optional[str],
        multi_node_configs: List[Dict],
        decision_method: str,
        mode: str = 'FullConnected',
        optimized_spatial: bool = False,
        initial_spatial_probability: float = 0.5,
        fixed_spatial_masks: List[int] = None,
        optimized_temporal: bool = False,
        initial_temporal_probability: float = 0.5,
        fixed_temporal_masks: List[int] = None,
    ):
        """
        Initialize the MultiNodeGraph.

        Args:
            domain: Domain context (e.g., "gsm8k", "humaneval")
            llm_name: Name of the LLM to use for all nodes
            multi_node_configs: List of configs for each MultiNode, each containing:
                - role: The role for this MultiNode (e.g., "Math Solver")
                - agent_name: Registered agent name for workers (default: "MathSolver")
                - num_agents: Number of workers per MultiNode (default: 3)
                - aggregator_method: Custom aggregator (default: "FinalMultiAgentAggregator")
            decision_method: Method for final decision aggregation
            mode: Topology mode for connections ('FullConnected', 'Chain', 'Debate', etc.)
            optimized_spatial: Whether spatial edges are learnable
            initial_spatial_probability: Initial probability for spatial edges
            fixed_spatial_masks: Fixed masks for spatial edges (1=enabled, 0=disabled)
            optimized_temporal: Whether temporal edges are learnable
            initial_temporal_probability: Initial probability for temporal edges
            fixed_temporal_masks: Fixed masks for temporal edges
        """
        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.multi_node_configs = multi_node_configs
        self.mode = mode
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal

        # Core components
        self.multi_nodes: Dict[str, MultiNode] = {}
        self.potential_spatial_edges: List[List[str]] = []
        self.potential_temporal_edges: List[List[str]] = []

        # Initialize prompt set first (needed for init_potential_edges)
        self.prompt_set = PromptSetRegistry.get(domain)

        # Initialize decision node
        self.decision_node: Node = AgentRegistry.get(
            decision_method,
            domain=self.domain,
            llm_name=self.llm_name
        )

        # Initialize MultiNodes and edges
        self.init_multi_nodes()
        self.init_potential_edges()

        # Handle case where no edges exist (single MultiNode or no connections)
        num_potential_edges = len(self.potential_spatial_edges)
        if num_potential_edges == 0:
            num_potential_edges = 1  # Avoid empty tensors

        # Generate masks based on mode if not provided
        if fixed_spatial_masks is None or fixed_temporal_masks is None:
            mode_spatial, mode_temporal = get_mode_masks(mode, self.num_multi_nodes)
            if fixed_spatial_masks is None:
                fixed_spatial_masks = mode_spatial
            if fixed_temporal_masks is None:
                fixed_temporal_masks = mode_temporal

        fixed_spatial_masks = torch.tensor(fixed_spatial_masks, dtype=torch.float)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks, dtype=torch.float)

        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks, requires_grad=False)
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks, requires_grad=False)

        # Initialize logits
        init_spatial_logit = (
            torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability)))
            if optimized_spatial else 10.0
        )
        self.spatial_logits = torch.nn.Parameter(
            torch.ones(num_potential_edges, requires_grad=optimized_spatial) * init_spatial_logit,
            requires_grad=optimized_spatial
        )

        init_temporal_logit = (
            torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability)))
            if optimized_temporal else 10.0
        )
        self.temporal_logits = torch.nn.Parameter(
            torch.ones(num_potential_edges, requires_grad=optimized_temporal) * init_temporal_logit,
            requires_grad=optimized_temporal
        )

        # GCN components for learnable edges (only if we have MultiNodes)
        if self.num_multi_nodes > 0:
            self.role_adj_matrix = self.construct_adj_matrix()
            self.features = self.construct_features()
            self.gcn = GCN(self.features.size(1) * 2, 16, self.features.size(1))
            self.mlp = MLP(1024, 16, 16)

    def init_multi_nodes(self):
        """Create MultiNode instances based on configs with heterogeneous LLM support."""
        for config in self.multi_node_configs:
            role = config.get('role', 'Math Solver')
            multi_node = MultiNode(
                id=f"mn_{role[:4]}_{shortuuid.ShortUUID().random(length=4)}",
                agent_name=config.get('agent_name', 'MathSolver'),
                num_agents=config.get('num_agents', 3),
                aggregator_method=config.get('aggregator_method', 'FinalMultiAgentAggregator'),
                domain=self.domain,
                llm_name=self.llm_name,
                role=role,
                worker_llm_names=config.get('worker_llm_names'),  # Per-worker LLM list
                aggregator_llm_name=config.get('aggregator_llm_name'),  # Aggregator LLM
            )
            self.add_multi_node(multi_node)

    def add_multi_node(self, multi_node: MultiNode) -> MultiNode:
        """Add a MultiNode to the graph with unique ID handling."""
        node_id = multi_node.id if multi_node.id else shortuuid.ShortUUID().random(length=4)
        while node_id in self.multi_nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        multi_node.id = node_id
        self.multi_nodes[node_id] = multi_node
        return multi_node

    def init_potential_edges(self):
        """Create all possible edges between MultiNode aggregators (N×N full connectivity).

        Unlike the original role-based approach, this creates potential edges for all
        MultiNode pairs. The actual connectivity is controlled by the mode parameter
        and masks, not by ROLE_CONNECTION. ROLE_CONNECTION is still used for GCN
        adjacency matrix (prior knowledge) via construct_adj_matrix().
        """
        for mn1_id in self.multi_nodes.keys():
            for mn2_id in self.multi_nodes.keys():
                self.potential_spatial_edges.append([mn1_id, mn2_id])
                self.potential_temporal_edges.append([mn1_id, mn2_id])

    @property
    def num_multi_nodes(self) -> int:
        """Return the number of MultiNodes in the graph."""
        return len(self.multi_nodes)

    @property
    def num_edges(self) -> int:
        """Count inter-MultiNode connections (not internal worker->aggregator)."""
        num_edges = 0
        for mn in self.multi_nodes.values():
            for succ in mn.aggregator_node.spatial_successors:
                # Only count if successor is another MultiNode's aggregator
                if self._find_multinode_by_aggregator(succ):
                    num_edges += 1
        return num_edges

    def find_multi_node(self, mn_id: str) -> MultiNode:
        """Find a MultiNode by its ID."""
        if mn_id in self.multi_nodes:
            return self.multi_nodes[mn_id]
        raise Exception(f"MultiNode not found: {mn_id}")

    def _find_multinode_by_aggregator(self, aggregator: Node) -> Optional[MultiNode]:
        """Find the MultiNode that owns a given aggregator node."""
        for mn in self.multi_nodes.values():
            if mn.aggregator_node == aggregator or mn.aggregator_node.id == aggregator.id:
                return mn
        return None

    def clear_spatial_connections(self):
        """Clear inter-MultiNode connections (keep internal worker→aggregator)."""
        for mn in self.multi_nodes.values():
            # Keep only internal workers as predecessors
            mn.aggregator_node.spatial_predecessors = [
                pred for pred in mn.aggregator_node.spatial_predecessors
                if pred in mn.worker_nodes
            ]
            # Clear successors that are other aggregators (keep decision node if connected)
            mn.aggregator_node.spatial_successors = []

        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []

    def clear_temporal_connections(self):
        """Clear all temporal connections between MultiNodes."""
        for mn in self.multi_nodes.values():
            mn.aggregator_node.temporal_predecessors = []
            mn.aggregator_node.temporal_successors = []
            for worker in mn.worker_nodes:
                worker.temporal_predecessors = []
                worker.temporal_successors = []

    def connect_decision_node(self):
        """Connect all MultiNode aggregators to the decision node."""
        for mn in self.multi_nodes.values():
            if self.decision_node not in mn.aggregator_node.spatial_successors:
                mn.aggregator_node.add_successor(self.decision_node, 'spatial')

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None):
        """
        Build spatial connections between MultiNode aggregators based on logits/masks.

        For each potential edge (src_mn_id, dst_mn_id):
        - Connect src_mn.aggregator_node as spatial predecessor of dst_mn.aggregator_node
        - This allows dst_mn's aggregator to see src_mn's synthesized output
        """
        self.clear_spatial_connections()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]

        if len(self.potential_spatial_edges) == 0:
            return torch.sum(torch.stack(log_probs))

        for i, potential_conn in enumerate(self.potential_spatial_edges):
            src_id, dst_id = potential_conn
            edge_logit = self.spatial_logits[i] if i < len(self.spatial_logits) else self.spatial_logits[0]
            edge_mask = self.spatial_masks[i] if i < len(self.spatial_masks) else self.spatial_masks[0]

            src_mn = self.multi_nodes[src_id]
            dst_mn = self.multi_nodes[dst_id]

            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and not self.optimized_spatial:
                # Fixed connection
                if not self.check_cycle(dst_mn, {src_mn}):
                    src_mn.aggregator_node.add_successor(dst_mn.aggregator_node, 'spatial')
                continue

            # Learnable connection
            if not self.check_cycle(dst_mn, {src_mn}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1.0 if edge_prob > threshold else 0.0)
                if torch.rand(1) < edge_prob:
                    src_mn.aggregator_node.add_successor(dst_mn.aggregator_node, 'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        return torch.sum(torch.stack(log_probs))

    def construct_temporal_connection(self, round: int = 0, temperature: float = 1.0, threshold: float = None):
        """Build temporal connections between MultiNode aggregators."""
        self.clear_temporal_connections()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]

        if round == 0 or len(self.potential_temporal_edges) == 0:
            return torch.sum(torch.stack(log_probs))

        for i, potential_conn in enumerate(self.potential_temporal_edges):
            src_id, dst_id = potential_conn
            edge_logit = self.temporal_logits[i] if i < len(self.temporal_logits) else self.temporal_logits[0]
            edge_mask = self.temporal_masks[i] if i < len(self.temporal_masks) else self.temporal_masks[0]

            src_mn = self.multi_nodes[src_id]
            dst_mn = self.multi_nodes[dst_id]

            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and not self.optimized_temporal:
                src_mn.aggregator_node.add_successor(dst_mn.aggregator_node, 'temporal')
                continue

            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1.0 if edge_prob > threshold else 0.0)
            if torch.rand(1) < edge_prob:
                src_mn.aggregator_node.add_successor(dst_mn.aggregator_node, 'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))

        return torch.sum(torch.stack(log_probs))

    def check_cycle(self, new_mn: MultiNode, target_mns: set) -> bool:
        """Check if adding new_mn creates a cycle with target MultiNodes."""
        if new_mn in target_mns:
            return True
        for succ in new_mn.aggregator_node.spatial_successors:
            succ_mn = self._find_multinode_by_aggregator(succ)
            if succ_mn and self.check_cycle(succ_mn, target_mns):
                return True
        return False

    def _compute_in_degrees(self) -> Dict[str, int]:
        """Compute in-degrees for MultiNodes based on external aggregator predecessors."""
        in_degree = {}
        for mn_id, mn in self.multi_nodes.items():
            # Count predecessors that are OTHER MultiNode aggregators (not own workers)
            external_preds = [
                pred for pred in mn.aggregator_node.spatial_predecessors
                if pred not in mn.worker_nodes
            ]
            in_degree[mn_id] = len(external_preds)
        return in_degree

    def update_memory(self):
        """Update memory for all nodes in all MultiNodes (for temporal connections)."""
        for mn in self.multi_nodes.values():
            for worker in mn.worker_nodes:
                worker.update_memory()
            mn.aggregator_node.update_memory()

    def construct_adj_matrix(self) -> torch.Tensor:
        """Construct adjacency matrix based on ROLE_CONNECTION for GCN."""
        role_connections = self.prompt_set.get_role_connection()
        num_multi_nodes = self.num_multi_nodes
        role_adj = torch.zeros((num_multi_nodes, num_multi_nodes))
        role_to_idx: Dict[str, List[int]] = {}

        for i, (mn_id, mn) in enumerate(self.multi_nodes.items()):
            if mn.role not in role_to_idx:
                role_to_idx[mn.role] = []
            role_to_idx[mn.role].append(i)

        for src_role, dst_role in role_connections:
            src_indices = role_to_idx.get(src_role, [])
            dst_indices = role_to_idx.get(dst_role, [])
            for src_idx in src_indices:
                for dst_idx in dst_indices:
                    if src_idx != dst_idx:
                        role_adj[src_idx][dst_idx] = 1

        edge_index, _ = dense_to_sparse(role_adj)
        return edge_index

    def construct_features(self) -> torch.Tensor:
        """Construct feature matrix for MultiNodes based on role descriptions."""
        features = []
        for mn_id, mn in self.multi_nodes.items():
            profile = self.prompt_set.get_description(mn.role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        return torch.tensor(np.array(features))

    def construct_new_features(self, query: str) -> torch.Tensor:
        """Construct query-enhanced features for GCN."""
        query_embedding = torch.tensor(get_sentence_embedding(query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_multi_nodes, 1))
        new_features = torch.cat((self.features, query_embedding), dim=1)
        return new_features

    def run(
        self,
        inputs: Dict[str, Any],
        num_rounds: int = 3,
        max_tries: int = 3,
        max_time: int = 600,
    ) -> Tuple[List[Any], torch.Tensor]:
        """
        Execute the MultiNodeGraph synchronously.

        Each round:
        1. Construct spatial connections between MultiNode aggregators
        2. Construct temporal connections (if applicable)
        3. Topologically sort MultiNodes based on aggregator dependencies
        4. Execute each MultiNode (workers -> aggregator)
        5. Update memory for temporal connections

        Finally, connect all aggregators to decision node and execute.

        Args:
            inputs: Input dict, typically {'task': '...'}
            num_rounds: Number of execution rounds
            max_tries: Maximum retry attempts per MultiNode
            max_time: Maximum time per execution (unused in sync mode)

        Returns:
            Tuple of (final_answers, log_probs)
        """
        log_probs = torch.tensor(0.0)

        for round in range(num_rounds):
            log_probs = log_probs + self.construct_spatial_connection()
            log_probs = log_probs + self.construct_temporal_connection(round)

            # Topological sort of MultiNodes
            in_degree = self._compute_in_degrees()
            queue = [mn_id for mn_id, deg in in_degree.items() if deg == 0]

            while queue:
                current_mn_id = queue.pop(0)
                current_mn = self.multi_nodes[current_mn_id]

                # Execute MultiNode (workers + aggregator)
                tries = 0
                while tries < max_tries:
                    try:
                        current_mn.execute(inputs)
                        break
                    except Exception as e:
                        print(f"Error in MultiNode {current_mn_id}: {e}")
                    tries += 1

                # Update successor in-degrees
                for succ_agg in current_mn.aggregator_node.spatial_successors:
                    succ_mn = self._find_multinode_by_aggregator(succ_agg)
                    if succ_mn and succ_mn.id in in_degree:
                        in_degree[succ_mn.id] -= 1
                        if in_degree[succ_mn.id] == 0:
                            queue.append(succ_mn.id)

            self.update_memory()

        # Final decision
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs

        if len(final_answers) == 0:
            final_answers.append("No answer from decision node")

        return final_answers, log_probs

    async def arun(
        self,
        inputs: Dict[str, Any],
        num_rounds: int = 3,
        max_tries: int = 3,
        max_time: int = 600,
    ) -> Tuple[List[Any], torch.Tensor]:
        """
        Execute the MultiNodeGraph asynchronously.

        Similar to run() but uses async execution for better performance.

        Args:
            inputs: Input dict, typically {'task': '...'}
            num_rounds: Number of execution rounds
            max_tries: Maximum retry attempts per MultiNode
            max_time: Timeout per MultiNode execution in seconds

        Returns:
            Tuple of (final_answers, log_probs)
        """
        log_probs = torch.tensor(0.0)

        # GCN-based edge prediction (if using learnable edges)
        if self.num_multi_nodes > 0 and self.optimized_spatial:
            new_features = self.construct_new_features(inputs['task'])
            logits = self.gcn(new_features, self.role_adj_matrix)
            logits = self.mlp(logits)
            self.spatial_logits = logits @ logits.t()
            self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

        for round in range(num_rounds):
            log_probs = log_probs + self.construct_spatial_connection()
            log_probs = log_probs + self.construct_temporal_connection(round)

            # Topological sort of MultiNodes
            in_degree = self._compute_in_degrees()
            queue = [mn_id for mn_id, deg in in_degree.items() if deg == 0]

            while queue:
                current_mn_id = queue.pop(0)
                current_mn = self.multi_nodes[current_mn_id]

                # Execute MultiNode asynchronously
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(
                            current_mn.async_execute(inputs),
                            timeout=max_time
                        )
                        break
                    except asyncio.TimeoutError:
                        print(f"Timeout in MultiNode {current_mn_id}")
                    except Exception as e:
                        print(f"Error in MultiNode {current_mn_id}: {e}")
                    tries += 1

                # Update successor in-degrees
                for succ_agg in current_mn.aggregator_node.spatial_successors:
                    succ_mn = self._find_multinode_by_aggregator(succ_agg)
                    if succ_mn and succ_mn.id in in_degree:
                        in_degree[succ_mn.id] -= 1
                        if in_degree[succ_mn.id] == 0:
                            queue.append(succ_mn.id)

            self.update_memory()

        # Final decision
        self.connect_decision_node()
        await self.decision_node.async_execute(inputs)
        final_answers = self.decision_node.outputs

        if len(final_answers) == 0:
            final_answers.append("No answer from decision node")

        return final_answers, log_probs

    def update_masks(self, pruning_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prune edges by zeroing masks based on logit values."""
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges * pruning_rate)
            if prune_num_edges < 1:
                prune_num_edges = 1

            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0

            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0

        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges * pruning_rate)
            if prune_num_edges < 1:
                prune_num_edges = 1

            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0

            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0

        return self.spatial_masks, self.temporal_masks

    @property
    def spatial_adj_matrix(self) -> np.ndarray:
        """Get the current spatial adjacency matrix between MultiNodes."""
        matrix = np.zeros((self.num_multi_nodes, self.num_multi_nodes))
        mn_ids = list(self.multi_nodes.keys())
        for i, mn1_id in enumerate(mn_ids):
            for j, mn2_id in enumerate(mn_ids):
                mn2 = self.multi_nodes[mn2_id]
                if mn2.aggregator_node in self.multi_nodes[mn1_id].aggregator_node.spatial_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self) -> np.ndarray:
        """Get the current temporal adjacency matrix between MultiNodes."""
        matrix = np.zeros((self.num_multi_nodes, self.num_multi_nodes))
        mn_ids = list(self.multi_nodes.keys())
        for i, mn1_id in enumerate(mn_ids):
            for j, mn2_id in enumerate(mn_ids):
                mn2 = self.multi_nodes[mn2_id]
                if mn2.aggregator_node in self.multi_nodes[mn1_id].aggregator_node.temporal_successors:
                    matrix[i, j] = 1
        return matrix

    def get_all_internal_nodes(self) -> List[Node]:
        """Get all internal nodes from all MultiNodes (workers + aggregators)."""
        nodes = []
        for mn in self.multi_nodes.values():
            nodes.extend(mn.get_internal_nodes())
        return nodes


def min_max_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to range [-1, 1]."""
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val == 0:
        return torch.zeros_like(tensor)
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
