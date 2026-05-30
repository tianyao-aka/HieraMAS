"""
Random graph generation utilities for Stage 1 training.

This module provides functions to generate random DAGs (Directed Acyclic Graphs)
with guaranteed connectivity for use in the pre-generated graph pool approach.
"""

import random
from typing import List, Optional

import torch


def generate_random_graph(
    num_nodes: int,
    min_density: float = 0.3,
    max_density: float = 0.7,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate a random DAG with guaranteed connectivity.

    Algorithm (Three-Step):
    1. Random permutation for topological order (ensures acyclicity)
    2. Forced coverage: spanning tree (ensures connectivity)
    3. Additional random edges (increases density to target range)

    Args:
        num_nodes: Number of nodes in the graph
        min_density: Minimum edge density (fraction of max possible edges)
        max_density: Maximum edge density (fraction of max possible edges)
        seed: Random seed for reproducibility (optional)

    Returns:
        Flattened adjacency matrix (N*N,) with 1s for edges, 0s otherwise.
        Index mapping: idx = j * N + i (destination j, source i, row-major order)
    """
    if seed is not None:
        random.seed(seed)

    if num_nodes < 2:
        # Single node or empty graph
        return torch.zeros(num_nodes * num_nodes)

    # Initialize adjacency matrix (N x N)
    adj = torch.zeros(num_nodes, num_nodes)

    # Step 1: Random permutation for topological order
    # This ensures any edge (perm[i] -> perm[j]) where i < j respects topological order
    perm = list(range(num_nodes))
    random.shuffle(perm)

    # Step 2: Forced coverage (spanning tree) - ensures connectivity
    # For each node (except first), connect from a random predecessor
    edges = set()
    for i in range(1, num_nodes):
        # Pick a random parent from nodes earlier in topological order
        parent_idx = random.randint(0, i - 1)
        src = perm[parent_idx]
        dst = perm[i]
        edges.add((src, dst))
        adj[dst, src] = 1  # adj[j, i] means edge from i to j

    # Step 3: Additional random edges to reach target density
    # Max possible edges in a DAG with topological order: N*(N-1)/2
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    target_density = random.uniform(min_density, max_density)
    target_edges = max(len(edges), int(target_density * max_possible_edges))
    target_edges = min(target_edges, max_possible_edges)  # Cap at max

    # Generate candidate edges (respecting topological order)
    candidate_edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            src = perm[i]
            dst = perm[j]
            if (src, dst) not in edges:
                candidate_edges.append((src, dst))

    # Randomly select additional edges
    random.shuffle(candidate_edges)
    additional_needed = target_edges - len(edges)
    for src, dst in candidate_edges[:additional_needed]:
        edges.add((src, dst))
        adj[dst, src] = 1

    # Flatten to 1D: idx = j * N + i (row-major order)
    return adj.view(-1)


def generate_graph_pool(
    num_graphs: int,
    num_nodes: int,
    min_density: float = 0.3,
    max_density: float = 0.7,
    ensure_diverse: bool = True,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Generate a pool of random graph candidates.

    Args:
        num_graphs: Number of graphs to generate (capped at 100)
        num_nodes: Number of nodes in each graph
        min_density: Minimum edge density for each graph
        max_density: Maximum edge density for each graph
        ensure_diverse: If True, reject duplicate graphs
        seed: Random seed for reproducibility (optional)

    Returns:
        List of flattened adjacency tensors, each of shape (N*N,)
    """
    if seed is not None:
        random.seed(seed)

    num_graphs = min(num_graphs, 100)  # Cap at 100 as per requirement

    pool = []
    seen_hashes = set() if ensure_diverse else None

    attempts = 0
    max_attempts = num_graphs * 10  # Prevent infinite loops

    while len(pool) < num_graphs and attempts < max_attempts:
        attempts += 1
        graph = generate_random_graph(num_nodes, min_density, max_density)

        if ensure_diverse:
            # Use tuple of values as hash key
            graph_hash = tuple(graph.tolist())
            if graph_hash in seen_hashes:
                continue
            seen_hashes.add(graph_hash)

        pool.append(graph)

    if len(pool) < num_graphs:
        print(f"Warning: Could only generate {len(pool)} diverse graphs "
              f"(requested {num_graphs})")

    return pool


def visualize_graph(adj_flat: torch.Tensor, num_nodes: int, node_names: List[str] = None) -> str:
    """
    Create a text visualization of a graph for debugging.

    Args:
        adj_flat: Flattened adjacency matrix (N*N,)
        num_nodes: Number of nodes
        node_names: Optional list of node names

    Returns:
        String representation of the graph
    """
    if node_names is None:
        node_names = [f"N{i}" for i in range(num_nodes)]

    adj = adj_flat.view(num_nodes, num_nodes)
    lines = ["Graph edges:"]

    for j in range(num_nodes):
        predecessors = []
        for i in range(num_nodes):
            if adj[j, i] == 1:
                predecessors.append(node_names[i])
        if predecessors:
            lines.append(f"  {', '.join(predecessors)} -> {node_names[j]}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("Testing random graph generation...")

    # Generate a single graph
    graph = generate_random_graph(5, seed=42)
    print(f"Single graph shape: {graph.shape}")
    print(visualize_graph(graph, 5))

    # Generate a pool
    pool = generate_graph_pool(10, 5, seed=42)
    print(f"\nGenerated pool of {len(pool)} graphs")
    for i, g in enumerate(pool[:3]):
        print(f"\nGraph {i}:")
        print(visualize_graph(g, 5))
