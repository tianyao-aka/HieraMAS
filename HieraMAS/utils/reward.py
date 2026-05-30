"""
Reward Functions for Learnable LLM Selection.

This module provides cost-aware reward functions for RL training
of the LLM selector network.
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass, field


def reward_v2(solved: bool, cost: float, cost_sensitivity: float = 300.0) -> float:
    """
    Cost-aware reward function for policy gradient training.

    Args:
        solved: Whether the task was solved correctly
        cost: API cost incurred (in dollars)
        cost_sensitivity: How much to penalize cost (higher = more cost-sensitive)
            - For solved=True: uses cost_sensitivity (default 500)
            - For solved=False: uses fixed k=300 for failure penalty

    Returns:
        Reward value: [0.5, 1] if solved, [-2.5, -1] if failed
    """
    if solved:
        # Lower cost = higher reward (approaching 1.0), clamped to min 0.5
        reward = math.exp(-cost_sensitivity * cost)
        return max(reward, 0.5)
    else:
        # Failure penalty: starts at -1 and grows exponentially with cost, clamped to min -2.5
        reward = -math.exp(300.0 * cost)
        return max(reward, -2.5)


def reward_simple(solved: bool, cost: float, cost_weight: float = 10.0) -> float:
    """
    Simple linear reward function.

    Args:
        solved: Whether the task was solved correctly
        cost: API cost incurred
        cost_weight: Weight for cost penalty

    Returns:
        Reward value
    """
    base_reward = 1.0 if solved else -1.0
    cost_penalty = cost * cost_weight
    return base_reward - cost_penalty


def reward_accuracy_only(solved: bool, cost: float) -> float:
    """
    Reward function that only cares about accuracy, ignores cost.

    Useful for ablation studies or when cost is not a concern.

    Args:
        solved: Whether the task was solved correctly
        cost: Ignored

    Returns:
        1.0 if solved, -1.0 otherwise
    """
    return 1.0 if solved else -1.0


@dataclass
class RewardTracker:
    """
    Tracks reward statistics for logging and analysis.

    Maintains running statistics of rewards, costs, and solve rates
    over a training run.
    """

    rewards: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    solved: List[bool] = field(default_factory=list)
    llm_selections: List[List[str]] = field(default_factory=list)

    def add(self, reward: float, cost: float, is_solved: bool,
            llm_selection: List[str] = None):
        """
        Record a single trial result.

        Args:
            reward: Computed reward value
            cost: API cost incurred
            is_solved: Whether the task was solved
            llm_selection: List of selected LLM names (optional)
        """
        self.rewards.append(reward)
        self.costs.append(cost)
        self.solved.append(is_solved)
        if llm_selection:
            self.llm_selections.append(llm_selection)

    def summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics.

        Returns:
            Dictionary with mean_reward, mean_cost, solve_rate, etc.
        """
        n = len(self.rewards)
        if n == 0:
            return {
                'mean_reward': 0.0,
                'mean_cost': 0.0,
                'solve_rate': 0.0,
                'total_cost': 0.0,
                'num_samples': 0,
            }

        return {
            'mean_reward': sum(self.rewards) / n,
            'mean_cost': sum(self.costs) / n,
            'solve_rate': sum(self.solved) / n,
            'total_cost': sum(self.costs),
            'num_samples': n,
            'std_reward': self._std(self.rewards),
            'std_cost': self._std(self.costs),
        }

    def _std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def get_llm_distribution(self) -> Dict[str, int]:
        """
        Get distribution of LLM selections.

        Returns:
            Dictionary mapping LLM names to selection counts
        """
        distribution: Dict[str, int] = {}
        for selection in self.llm_selections:
            for llm in selection:
                distribution[llm] = distribution.get(llm, 0) + 1
        return distribution

    def reset(self):
        """Clear all tracked data."""
        self.rewards.clear()
        self.costs.clear()
        self.solved.clear()
        self.llm_selections.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class BaselineRewardNormalizer:
    """
    Normalizes rewards using a running baseline for variance reduction.

    Uses exponential moving average to track baseline reward.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: EMA decay factor (higher = faster adaptation)
        """
        self.alpha = alpha
        self.baseline = 0.0
        self.initialized = False

    def normalize(self, reward: float) -> float:
        """
        Normalize reward by subtracting baseline.

        Args:
            reward: Raw reward value

        Returns:
            Normalized reward (reward - baseline)
        """
        if not self.initialized:
            self.baseline = reward
            self.initialized = True
            return 0.0

        normalized = reward - self.baseline
        # Update baseline with EMA
        self.baseline = self.alpha * reward + (1 - self.alpha) * self.baseline
        return normalized

    def reset(self):
        """Reset baseline."""
        self.baseline = 0.0
        self.initialized = False
