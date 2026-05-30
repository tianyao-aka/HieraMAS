"""
Stage 2 Classifier Training Script for HumanEval.

This script trains a GNN classifier to predict which graph topology works best
for a given HumanEval coding task. The classifier is trained on labeled data
from Stage 1 execution.

Training approach:
- Split by TASK (not by graph) to ensure generalization to new tasks
- Binary classification with BCE loss
- Class weighting to handle imbalanced data
- Early stopping based on validation performance

Usage:
    python experiments_v2/humaneval/run_stage2_classifier_humaneval.py \
        --dataset_file result_v2/humaneval/stage2/humaneval_stage2_dataset.pkl \
        --output_dir result_v2/humaneval/stage2/classifier \
        --num_epochs 100 \
        --batch_size 32

Output:
    - stage2_classifier_best.pt: Best model checkpoint
    - training_log.json: Training metrics
"""

import os
os.environ['TRUST_REMOTE_CODE'] = 'true'
os.environ['HF_HUB_TRUST_REMOTE_CODE'] = '1'

import sys
import argparse
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

from HieraMAS.model.graph_classifier import GraphClassifier


# Role descriptions for HumanEval agents (for pre-trained node embeddings)
ROLE_DESCRIPTIONS = [
    "Project Manager: Coordinates the team, breaks down requirements, and ensures code meets specifications.",
    "Algorithm Designer: Designs efficient algorithms and data structures for solving programming problems.",
    "Programming Expert: Implements clean, efficient code following best practices and design patterns.",
    "Test Analyst: Designs test cases, identifies edge cases, and validates code correctness.",
    "Bug Fixer: Debugs code, identifies errors, and fixes issues to ensure correct functionality.",
]


class GraphDataset(Dataset):
    """
    PyTorch Dataset for graph classification.

    Each sample contains:
    - task_emb: Pre-computed task embedding (1024,)
    - graph: Flattened adjacency matrix (25,) for 5×5
    - label: Binary label (1 = good graph, 0 = bad graph)
    - task: Task string for grouping
    """

    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize the dataset.

        Args:
            data: List of dicts with 'task_emb', 'graph', 'label' keys
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            'task_emb': item['task_emb'].clone(),
            'graph': item['graph'].clone(),
            'label': torch.tensor(item['label'], dtype=torch.float32),
            'task': item.get('task', ''),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for batching."""
    return {
        'task_embs': torch.stack([item['task_emb'] for item in batch]),
        'graphs': torch.stack([item['graph'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
        'tasks': [item['task'] for item in batch],
    }


def split_by_task(
    data: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data by task to ensure no task overlap between splits.

    This is important for proper evaluation - we want to test generalization
    to new tasks, not memorization of seen tasks.

    Args:
        data: Full dataset
        train_ratio: Fraction of tasks for training (rest goes to validation)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data)
    """
    random.seed(seed)

    # Group samples by task
    task_to_samples = defaultdict(list)
    for item in data:
        task_to_samples[item['task']].append(item)

    # Get unique tasks and shuffle
    tasks = list(task_to_samples.keys())
    random.shuffle(tasks)

    # Split tasks (80/20 train/val)
    n_train = int(len(tasks) * train_ratio)

    train_tasks = set(tasks[:n_train])
    val_tasks = set(tasks[n_train:])

    # Collect samples for each split
    train_data = [item for item in data if item['task'] in train_tasks]
    val_data = [item for item in data if item['task'] in val_tasks]

    return train_data, val_data


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        threshold: Classification threshold

    Returns:
        Dict with accuracy, precision, recall, f1
    """
    pred_binary = (predictions >= threshold).astype(int)
    labels_binary = labels.astype(int)

    # Accuracy
    accuracy = (pred_binary == labels_binary).mean()

    # True/False positives/negatives
    tp = ((pred_binary == 1) & (labels_binary == 1)).sum()
    fp = ((pred_binary == 1) & (labels_binary == 0)).sum()
    fn = ((pred_binary == 0) & (labels_binary == 1)).sum()
    tn = ((pred_binary == 0) & (labels_binary == 0)).sum()

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }


def compute_per_task_roc(
    predictions: List[float],
    labels: List[int],
    tasks: List[str],
) -> Dict[str, float]:
    """
    Compute per-task ROC-AUC and return mean/std across tasks.

    For each task (which has multiple graph samples), compute the ROC-AUC
    of distinguishing good graphs (label=1) from bad graphs (label=0).
    This better reflects real-world performance than global metrics.

    Args:
        predictions: List of predicted probabilities
        labels: List of ground truth labels (0 or 1)
        tasks: List of task strings (same length as predictions/labels)

    Returns:
        Dict with mean_roc_auc, std_roc_auc, num_tasks_evaluated
    """
    # Group by task
    task_to_preds = defaultdict(list)
    task_to_labels = defaultdict(list)

    for pred, label, task in zip(predictions, labels, tasks):
        task_to_preds[task].append(pred)
        task_to_labels[task].append(label)

    # Compute ROC-AUC for each task
    roc_scores = []
    for task in task_to_preds:
        preds = task_to_preds[task]
        lbls = task_to_labels[task]

        # Need at least one positive and one negative sample to compute ROC
        if len(set(lbls)) < 2:
            continue

        try:
            roc = roc_auc_score(lbls, preds)
            roc_scores.append(roc)
        except ValueError:
            # Skip if ROC cannot be computed
            continue

    if len(roc_scores) == 0:
        return {
            'mean_roc_auc': 0.0,
            'std_roc_auc': 0.0,
            'num_tasks_evaluated': 0,
        }

    return {
        'mean_roc_auc': float(np.mean(roc_scores)),
        'std_roc_auc': float(np.std(roc_scores)),
        'num_tasks_evaluated': len(roc_scores),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        task_embs = batch['task_embs'].to(device)
        graphs = batch['graphs'].to(device)
        labels = batch['labels'].to(device)
        tasks = batch['tasks']

        optimizer.zero_grad()

        # Forward pass (batch)
        batch_logits = []
        for i in range(len(tasks)):
            logit = model.forward(tasks[i], graphs[i], task_embs[i])
            batch_logits.append(logit)
        logits = torch.stack(batch_logits).squeeze(-1)

        # Compute loss
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Collect predictions for metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            all_predictions.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))

    return avg_loss, metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Evaluate model on a dataset.

    Returns:
        Tuple of (average_loss, metrics_dict, roc_metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_tasks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            task_embs = batch['task_embs'].to(device)
            graphs = batch['graphs'].to(device)
            labels = batch['labels'].to(device)
            tasks = batch['tasks']

            # Forward pass (batch)
            batch_logits = []
            for i in range(len(tasks)):
                logit = model.forward(tasks[i], graphs[i], task_embs[i])
                batch_logits.append(logit)
            logits = torch.stack(batch_logits).squeeze(-1)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Collect predictions and tasks
            probs = torch.sigmoid(logits).cpu().numpy()
            all_predictions.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_tasks.extend(tasks)

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    roc_metrics = compute_per_task_roc(all_predictions, all_labels, all_tasks)

    return avg_loss, metrics, roc_metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 2 Classifier Training for HumanEval"
    )

    # Data
    parser.add_argument(
        '--dataset_file',
        type=str,
        required=True,
        help='Path to Stage 2 dataset pickle file'
    )

    # Model
    parser.add_argument(
        '--gcn_hidden_dim',
        type=int,
        default=256,
        help='GCN hidden dimension'
    )
    parser.add_argument(
        '--gcn_output_dim',
        type=int,
        default=128,
        help='GCN output dimension'
    )
    parser.add_argument(
        '--classifier_hidden_dim',
        type=int,
        default=64,
        help='Classifier MLP hidden dimension'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate'
    )

    # Training
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay (L2 regularization)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (epochs without improvement)'
    )

    # Data splitting
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Fraction of tasks for training (rest goes to validation)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='result_v2/humaneval/stage2/classifier',
        help='Output directory'
    )

    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset_file}")
    with open(args.dataset_file, 'rb') as f:
        dataset_dict = pickle.load(f)

    # Handle both checkpoint formats:
    # Format 1: {'data': [...], 'metadata': {...}}
    # Format 2: {'all_data': [...], 'total_positive': int, 'total_negative': int, 'args': {...}}
    if 'data' in dataset_dict:
        data = dataset_dict['data']
        metadata = dataset_dict['metadata']
    elif 'all_data' in dataset_dict:
        data = dataset_dict['all_data']
        metadata = {
            'total_positive': dataset_dict.get('total_positive', 'N/A'),
            'total_negative': dataset_dict.get('total_negative', 'N/A'),
            'args': dataset_dict.get('args', {}),
        }
    else:
        raise KeyError(f"Unknown dataset format. Keys: {dataset_dict.keys()}")

    num_nodes = metadata.get('num_nodes', 5)  # Default 5 for HumanEval

    print(f"Total samples: {len(data)}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Positive samples: {metadata.get('total_positive', 'N/A')}")
    print(f"Negative samples: {metadata.get('total_negative', 'N/A')}")

    # Split by task (80/20 train/val)
    print(f"\nSplitting data by task (train={args.train_ratio}, val={1-args.train_ratio:.2f})...")
    train_data, val_data = split_by_task(data, args.train_ratio, args.seed)
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Create datasets and dataloaders
    train_dataset = GraphDataset(train_data)
    val_dataset = GraphDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Compute class weights for imbalanced data
    train_labels = [item['label'] for item in train_data]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0], device=device)
    print(f"\nClass weights: pos_weight={pos_weight.item():.2f}")

    # Initialize model with pre-trained role embeddings
    model = GraphClassifier(
        num_nodes=num_nodes,
        gcn_hidden_dim=args.gcn_hidden_dim,
        gcn_output_dim=args.gcn_output_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        role_descriptions=ROLE_DESCRIPTIONS,
    )
    model.to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    best_val_roc = 0.0
    best_epoch = 0
    patience_counter = 0
    training_log = []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate (now returns 3 values including ROC metrics)
        val_loss, val_metrics, val_roc = evaluate(model, val_loader, criterion, device)

        # Update scheduler based on ROC-AUC (better metric for ranking)
        scheduler.step(val_roc['mean_roc_auc'])

        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'val_roc': val_roc,
            'lr': optimizer.param_groups[0]['lr'],
        }
        training_log.append(log_entry)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, ROC-AUC: {val_roc['mean_roc_auc']:.4f}")

        # Save best model (use ROC-AUC as primary metric)
        if val_roc['mean_roc_auc'] > best_val_roc:
            best_val_roc = val_roc['mean_roc_auc']
            best_epoch = epoch + 1
            patience_counter = 0

            best_path = output_dir / "stage2_classifier_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'val_roc': val_roc,
                'args': vars(args),
                'metadata': metadata,
            }, best_path)
            print(f"  Saved best model (ROC-AUC: {best_val_roc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
            break

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)

    # Final evaluation on validation set with best model
    print("\n" + "=" * 80)
    print("Final Evaluation on Validation Set (Best Model)")
    print("=" * 80)

    # Load best model
    best_path = output_dir / "stage2_classifier_best.pt"
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_metrics, final_roc = evaluate(model, val_loader, criterion, device)

    print(f"Val Loss: {final_loss:.4f}")
    print(f"Val Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Val Precision: {final_metrics['precision']:.4f}")
    print(f"Val Recall: {final_metrics['recall']:.4f}")
    print(f"Val F1: {final_metrics['f1']:.4f}")
    print(f"Val ROC-AUC: {final_roc['mean_roc_auc']:.4f} (+/- {final_roc['std_roc_auc']:.4f})")
    print(f"Tasks evaluated: {final_roc['num_tasks_evaluated']}")

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'metadata': {
                'num_nodes': num_nodes,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
            },
            'best_epoch': best_epoch,
            'best_val_roc_auc': best_val_roc,
            'final_val_metrics': final_metrics,
            'final_val_roc': final_roc,
            'training_log': training_log,
        }, f, indent=2)
    print(f"\nTraining log saved to: {log_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Validation ROC-AUC: {best_val_roc:.4f} (epoch {best_epoch})")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
