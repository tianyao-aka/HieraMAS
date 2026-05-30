"""
Graph Classifier for Stage 2 Training.

This module implements a GNN-based classifier that predicts whether a graph
topology is suitable for a given task. The classifier is trained on labeled
data generated from Stage 1 execution.

Architecture:
    Task -> SentenceTransformer -> task_emb (1024)
                                       |
                                       v
    Node features + task_emb -> [concat] -> (N, 2048)
                                       |
                                       v
    Graph edges -> SimpleGCN -> node_emb (N, 128)
                                       |
                                       v
                   global_mean_pool -> graph_emb (128)
                                       |
                                       v
                        MLP Classifier -> logit (1)
                                       |
                                       v
                              sigmoid -> P(good graph)
"""

import torch
import torch.nn as nn
from typing import Optional, List
from torch_geometric.utils import dense_to_sparse

from HieraMAS.gnn.simple_gcn import SimpleGCN
from HieraMAS.llm.profile_embedding import get_sentence_embedding


class GraphClassifier(nn.Module):
    """
    GNN-based classifier for predicting graph quality given a task.

    The model takes a task string and a graph adjacency matrix, and outputs
    a probability that the graph is a good fit for the task.
    """

    def __init__(
        self,
        node_feature_dim: int = 1024,
        task_embed_dim: int = 1024,
        gcn_hidden_dim: int = 256,
        gcn_output_dim: int = 128,
        classifier_hidden_dim: int = 64,
        num_nodes: int = 8,
        dropout: float = 0.2,
        role_descriptions: Optional[List[str]] = None,
    ):
        """
        Initialize the GraphClassifier.

        Args:
            node_feature_dim: Dimension of node feature embeddings (role embeddings)
            task_embed_dim: Dimension of task embeddings from sentence transformer
            gcn_hidden_dim: Hidden dimension for GCN (intermediate processing)
            gcn_output_dim: Output dimension of GCN node embeddings
            classifier_hidden_dim: Hidden dimension of the final MLP classifier
            num_nodes: Number of nodes in the graph (default 8 for 8 agents)
            dropout: Dropout rate for regularization
            role_descriptions: Optional list of role description strings for pre-trained
                              node embeddings. If provided, sentence embeddings will be
                              computed and used as initial node features.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.task_embed_dim = task_embed_dim

        # Input to GCN: node_features (1024) + task_emb (1024) = 2048
        gcn_input_dim = node_feature_dim + task_embed_dim

        # SimpleGCN for graph-level representation
        self.gcn = SimpleGCN(input_dim=gcn_input_dim, output_dim=gcn_output_dim)

        # MLP classifier: graph_emb -> logit
        self.classifier = nn.Sequential(
            nn.Linear(gcn_output_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

        # Initialize node features (role embeddings)
        if role_descriptions is not None:
            # Use pre-trained sentence embeddings from role descriptions
            print(f"Initializing node features from {len(role_descriptions)} role descriptions...")
            features = []
            for desc in role_descriptions:
                emb = get_sentence_embedding(desc)
                features.append(torch.tensor(emb, dtype=torch.float32))
            init_features = torch.stack(features)
            assert init_features.shape[0] == num_nodes, \
                f"Number of role descriptions ({init_features.shape[0]}) must match num_nodes ({num_nodes})"
        else:
            # Fall back to random initialization
            init_features = torch.randn(num_nodes, node_feature_dim) * 0.02

        self.node_features = nn.Parameter(init_features)

        # Cache for task embeddings
        self._task_emb_cache = {}

    def get_task_embedding(
        self,
        task: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get sentence embedding for a task string.

        Args:
            task: Task text string
            device: Target device for tensor

        Returns:
            Task embedding tensor of shape (task_embed_dim,)
        """
        # Use hash for cache key (truncate task to avoid memory issues)
        task_truncated = task[:2000]
        cache_key = hash(task_truncated)

        if cache_key not in self._task_emb_cache:
            emb = get_sentence_embedding(task_truncated)
            self._task_emb_cache[cache_key] = torch.tensor(emb, dtype=torch.float32)

        return self._task_emb_cache[cache_key].to(device)

    def clear_cache(self):
        """Clear the task embedding cache."""
        self._task_emb_cache.clear()

    def forward(
        self,
        task: str,
        graph_adj: torch.Tensor,
        task_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict P(good graph | task, graph).

        Args:
            task: Task text string
            graph_adj: Flattened adjacency matrix (N*N,) or 2D (N, N)
            task_emb: Optional pre-computed task embedding (1024,). If None,
                      computed from task string.

        Returns:
            Logit tensor of shape (1,) - apply sigmoid to get probability
        """
        device = self.node_features.device

        # Reshape adjacency if needed
        if graph_adj.dim() == 1:
            adj_matrix = graph_adj.view(self.num_nodes, self.num_nodes)
        else:
            adj_matrix = graph_adj
        adj_matrix = adj_matrix.to(device)

        # Get task embedding
        if task_emb is None:
            task_emb = self.get_task_embedding(task, device)
        else:
            task_emb = task_emb.to(device)

        # Broadcast task embedding to all nodes: (N, task_embed_dim)
        task_emb_broadcast = task_emb.unsqueeze(0).expand(self.num_nodes, -1)

        # Concatenate node features with task embedding: (N, node_feat + task_emb)
        node_input = torch.cat([self.node_features, task_emb_broadcast], dim=-1)

        # Convert adjacency matrix to edge_index format for GCN
        edge_index, _ = dense_to_sparse(adj_matrix)

        # GCN forward: (N, gcn_input_dim) -> (N, gcn_output_dim)
        node_emb = self.gcn(node_input, edge_index)

        # Global mean pooling: (N, gcn_output_dim) -> (gcn_output_dim,)
        graph_emb = node_emb.mean(dim=0)

        # Classifier: (gcn_output_dim,) -> (1,)
        logit = self.classifier(graph_emb)

        return logit

    def forward_batch(
        self,
        tasks: List[str],
        graph_adjs: torch.Tensor,
        task_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch forward pass for efficiency.

        Args:
            tasks: List of task strings (batch_size,)
            graph_adjs: Batch of flattened adjacency matrices (batch_size, N*N)
            task_embs: Optional pre-computed task embeddings (batch_size, 1024)

        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        device = self.node_features.device
        batch_size = len(tasks)

        logits = []
        for i in range(batch_size):
            task_emb = task_embs[i] if task_embs is not None else None
            logit = self.forward(tasks[i], graph_adjs[i], task_emb)
            logits.append(logit)

        return torch.stack(logits)

    def forward_batch_efficient(
        self,
        task: str,
        graph_adjs: torch.Tensor,
        task_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch forward for multiple graphs sharing the same task.

        This is more efficient than forward_batch when all graphs share the same task,
        as it avoids redundant task embedding computation and reuses node_input.

        Args:
            task: Task text string (shared across all graphs)
            graph_adjs: Stacked adjacency matrices (batch_size, N*N)
            task_emb: Optional pre-computed task embedding (1024,)

        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        device = self.node_features.device
        batch_size = graph_adjs.size(0)
        graph_adjs = graph_adjs.to(device)
        adj_matrices = graph_adjs.view(batch_size, self.num_nodes, self.num_nodes)

        # Get task embedding (computed once, shared across all graphs)
        if task_emb is None:
            task_emb = self.get_task_embedding(task, device)
        else:
            task_emb = task_emb.to(device)

        # Prepare node input (shared across all graphs)
        task_emb_broadcast = task_emb.unsqueeze(0).expand(self.num_nodes, -1)
        node_input = torch.cat([self.node_features, task_emb_broadcast], dim=-1)

        # Process each graph
        logits = []
        for i in range(batch_size):
            edge_index, _ = dense_to_sparse(adj_matrices[i])
            node_emb = self.gcn(node_input, edge_index)
            graph_emb = node_emb.mean(dim=0)
            logits.append(self.classifier(graph_emb))

        return torch.stack(logits)

    def predict_proba(
        self,
        task: str,
        graph_adj: torch.Tensor,
        task_emb: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Predict probability that the graph is good for the task.

        Args:
            task: Task text string
            graph_adj: Flattened adjacency matrix (N*N,)
            task_emb: Optional pre-computed task embedding

        Returns:
            Probability in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            logit = self.forward(task, graph_adj, task_emb)
            prob = torch.sigmoid(logit).item()
        return prob

    def select_best_graph(
        self,
        task: str,
        candidate_graphs: List[torch.Tensor],
        task_emb: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Select the best graph from candidates for a given task.

        Args:
            task: Task text string
            candidate_graphs: List of flattened adjacency matrices
            task_emb: Optional pre-computed task embedding

        Returns:
            Tuple of (best_graph, best_score, all_scores)
        """
        self.eval()
        with torch.no_grad():
            scores = []
            for graph in candidate_graphs:
                logit = self.forward(task, graph, task_emb)
                score = torch.sigmoid(logit).item()
                scores.append(score)

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidate_graphs[best_idx], scores[best_idx], scores


def create_graph_classifier(
    num_nodes: int = 8,
    pretrained_node_features: Optional[torch.Tensor] = None,
) -> GraphClassifier:
    """
    Factory function to create a GraphClassifier with optional pre-trained node features.

    Args:
        num_nodes: Number of nodes in the graph
        pretrained_node_features: Optional pre-trained node feature embeddings
                                  of shape (num_nodes, node_feature_dim)

    Returns:
        Initialized GraphClassifier
    """
    classifier = GraphClassifier(num_nodes=num_nodes)

    if pretrained_node_features is not None:
        assert pretrained_node_features.shape[0] == num_nodes
        classifier.node_features.data = pretrained_node_features.clone()

    return classifier
