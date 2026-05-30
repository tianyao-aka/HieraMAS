"""
SimpleGCN: Parameter-free mean aggregation GCN for edge learning.

This module implements a simple 2-layer GCN that uses mean aggregation
without learnable parameters in the message passing layers. Only the
final projection layer has trainable parameters.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree


def soft_clamp(x: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    """
    Soft clamp using tanh that always maintains non-zero gradients.

    Unlike torch.clamp which has gradient=0 outside the range,
    soft_clamp always has gradient > 0, preventing gradient death
    when logits explode.

    The default limit of 10.0 ensures sigmoid has meaningful gradient:
    - At logit=5, sigmoid gradient ≈ 0.007
    - At logit=10, sigmoid gradient ≈ 0.00005 (small but non-zero)
    - At logit=20+, sigmoid gradient → 0 (gradient death)

    Args:
        x: Input tensor
        limit: Approximate clamping range [-limit, limit]. Default 10.0 keeps
               logits in a range where sigmoid has small but non-zero gradient.

    Returns:
        Tensor with values smoothly constrained to approximately [-limit, limit]
    """
    return torch.tanh(x / limit) * limit


class SimpleGCN(nn.Module):
    """
    A simple 2-layer GCN with mean aggregation and no learnable message passing.

    Architecture:
        Layer 1: Mean aggregation over neighbors
        Layer 2: Mean aggregation over neighbors
        Projection: Linear layer to output_dim

    The only trainable parameters are in the final projection layer.
    """

    def __init__(self, input_dim: int, output_dim: int = 128):
        """
        Initialize SimpleGCN.

        Args:
            input_dim: Dimension of input node features (node_feat + query_feat)
            output_dim: Dimension of output node embeddings
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Only learnable component: final projection
        self.projection = nn.Linear(input_dim, output_dim)

    def mean_aggregate(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Perform mean aggregation over neighbors.

        Args:
            x: Node features (N, D)
            edge_index: Edge indices (2, E) in COO format

        Returns:
            Aggregated features (N, D)
        """
        num_nodes = x.size(0)

        # Add self-loops so each node includes itself in aggregation
        edge_index_with_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Compute degree for normalization
        row, col = edge_index_with_self
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0  # Handle isolated nodes

        # Aggregate: sum neighbor features then normalize by degree
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row])
        out = out * deg_inv.unsqueeze(-1)

        return out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 2-layer mean aggregation + linear projection.

        Args:
            x: Node features (N, input_dim), already concatenated [node_feat || query_feat]
            edge_index: Edge indices (2, E) from prior adjacency matrix

        Returns:
            Node embeddings (N, output_dim)
        """
        # Layer 1: Mean aggregation
        h = self.mean_aggregate(x, edge_index)
        h = torch.relu(h)

        # Layer 2: Mean aggregation
        h = self.mean_aggregate(h, edge_index)
        h = torch.relu(h)

        # Final projection to output dimension
        out = self.projection(h)

        return out


class EdgeScorer(nn.Module):
    """
    MLP that scores edges based on source and destination node embeddings.

    Takes concatenated [src_emb || dst_emb] and outputs a scalar logit.
    """

    def __init__(self, node_emb_dim: int = 128, hidden_dim: int = 64):
        """
        Initialize EdgeScorer.

        Args:
            node_emb_dim: Dimension of node embeddings from SimpleGCN
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_emb: torch.Tensor, dst_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute edge logit for a single edge.

        Args:
            src_emb: Source node embedding (node_emb_dim,)
            dst_emb: Destination node embedding (node_emb_dim,)

        Returns:
            Scalar logit
        """
        pair_feat = torch.cat([src_emb, dst_emb], dim=-1)
        return self.mlp(pair_feat).squeeze(-1)

    def forward_all(self, node_emb: torch.Tensor, apply_clamp: bool = True) -> torch.Tensor:
        """
        Compute logits for all N*N possible edges.

        Args:
            node_emb: All node embeddings (N, node_emb_dim)
            apply_clamp: If True, apply soft_clamp to logits. Default True for training.
                        Set to False to get raw logits for inspection.

        Returns:
            Edge logits (N*N,) in row-major order [dst=0,src=0], [dst=0,src=1], ...
        """
        N = node_emb.size(0)
        logits = []

        for j in range(N):  # destination
            for i in range(N):  # source
                logit = self.forward(node_emb[i], node_emb[j])
                logits.append(logit)

        logits = torch.stack(logits)

        if apply_clamp:
            # Use soft clamp to prevent gradient death from logit explosion.
            # Hard clamp (torch.clamp) has gradient=0 outside the range,
            # but soft clamp (tanh-based) always has non-zero gradient.
            # Default limit=10.0 keeps logits where sigmoid has meaningful gradient.
            logits = soft_clamp(logits)

        return logits
