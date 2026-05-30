"""
GNN modules for HieraMAS.

This package contains Graph Neural Network implementations for learning
graph structures and edge predictions.
"""

from HieraMAS.gnn.gcn import GCN, MLP
from HieraMAS.gnn.simple_gcn import SimpleGCN, EdgeScorer

__all__ = ['GCN', 'MLP', 'SimpleGCN', 'EdgeScorer']
