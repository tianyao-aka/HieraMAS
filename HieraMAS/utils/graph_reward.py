"""
Graph-Level Reward Utilities for LearnableMultiNodeGraph.

This module provides reward computation and tracking utilities for
training graphs of LearnableMultiNodes with mixed reward strategies.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import math

from .reward import RewardTracker


def compute_mixed_reward(
    node_reward: float,
    final_reward: float,
    alpha: float = 0.5,
) -> float:
    """
    Compute effective reward using mixed strategy.

    The mixed reward balances individual node performance with overall
    graph performance:
    - alpha = 1.0: Only care about this node's output (local optimization)
    - alpha = 0.0: Only care about final output (global optimization)
    - alpha = 0.5: Balanced (recommended starting point)

    Args:
        node_reward: Reward for this specific node's output
        final_reward: Reward for the graph's final output
        alpha: Mixing coefficient (0 = only final, 1 = only node)

    Returns:
        Effective reward = alpha * node_reward + (1-alpha) * final_reward
    """
    return alpha * node_reward + (1 - alpha) * final_reward


@dataclass
class RoleRewardTracker:
    """
    Track reward statistics for a single role/node.

    Maintains per-role statistics including rewards, costs, solve rates,
    and LLM selection distributions.
    """

    role: str
    rewards: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    solved: List[bool] = field(default_factory=list)
    llm_selections: List[List[str]] = field(default_factory=list)
    effective_rewards: List[float] = field(default_factory=list)

    def add(
        self,
        reward: float,
        cost: float,
        is_solved: bool,
        llm_selection: Optional[List[str]] = None,
        effective_reward: Optional[float] = None,
    ):
        """
        Record a single trial result for this role.

        Args:
            reward: Per-node reward value
            cost: API cost incurred
            is_solved: Whether the node's output was correct
            llm_selection: List of selected LLM names
            effective_reward: Mixed reward (node + final)
        """
        self.rewards.append(reward)
        self.costs.append(cost)
        self.solved.append(is_solved)
        if llm_selection:
            self.llm_selections.append(llm_selection)
        if effective_reward is not None:
            self.effective_rewards.append(effective_reward)

    def summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for this role.

        Returns:
            Dictionary with mean_reward, mean_cost, solve_rate, etc.
        """
        n = len(self.rewards)
        if n == 0:
            return {
                'role': self.role,
                'mean_reward': 0.0,
                'mean_cost': 0.0,
                'solve_rate': 0.0,
                'mean_effective_reward': 0.0,
                'num_samples': 0,
            }

        return {
            'role': self.role,
            'mean_reward': sum(self.rewards) / n,
            'mean_cost': sum(self.costs) / n,
            'solve_rate': sum(self.solved) / n,
            'total_cost': sum(self.costs),
            'num_samples': n,
            'std_reward': self._std(self.rewards),
            'mean_effective_reward': (
                sum(self.effective_rewards) / len(self.effective_rewards)
                if self.effective_rewards else 0.0
            ),
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
        Get distribution of LLM selections for this role.

        Returns:
            Dictionary mapping LLM names to selection counts
        """
        distribution: Dict[str, int] = {}
        for selection in self.llm_selections:
            for llm in selection:
                distribution[llm] = distribution.get(llm, 0) + 1
        return distribution

    def get_latest_llm_selection(self) -> Optional[List[str]]:
        """
        Get the LLM selection from the latest round.

        Returns:
            List of LLM names selected in the most recent round, or None if no selections
        """
        if self.llm_selections:
            return self.llm_selections[-1]
        return None

    def reset(self):
        """Clear all tracked data."""
        self.rewards.clear()
        self.costs.clear()
        self.solved.clear()
        self.llm_selections.clear()
        self.effective_rewards.clear()


class GraphRewardTracker:
    """
    Track reward statistics for LearnableMultiNodeGraph training.

    Maintains per-role statistics as well as graph-level (final) statistics.
    Also tracks LLM selection distributions for each role.

    Usage:
        tracker = GraphRewardTracker()
        for result in training_results:
            tracker.add(result)
        summary = tracker.summary()
    """

    def __init__(self):
        """Initialize the tracker."""
        self.role_trackers: Dict[str, RoleRewardTracker] = {}
        self.final_tracker = RewardTracker()
        self._total_samples = 0
        # Skip learning tracking
        self.skip_counts: Dict[str, int] = {}  # {role: count of times skipped}
        self._skip_samples = 0  # Total samples for skip rate computation

    def add(self, forward_result: Dict[str, Any]) -> None:
        """
        Record results from one forward pass.

        Args:
            forward_result: Output from LearnableMultiNodeGraph.forward(),
                containing:
                - 'final_output': str
                - 'final_reward': float
                - 'final_solved': bool
                - 'node_results': Dict[role, {output, reward, solved, cost, llm_selection}]
                - 'effective_rewards': Dict[role, float]
                - 'total_cost': float
        """
        # Track final result
        final_reward = forward_result.get('final_reward', 0.0)
        final_cost = forward_result.get('total_cost', 0.0)
        final_solved = forward_result.get('final_solved', False)
        self.final_tracker.add(final_reward, final_cost, final_solved)

        # Track per-role results
        node_results = forward_result.get('node_results', {})
        effective_rewards = forward_result.get('effective_rewards', {})

        for role, result in node_results.items():
            if role not in self.role_trackers:
                self.role_trackers[role] = RoleRewardTracker(role=role)

            tracker = self.role_trackers[role]
            tracker.add(
                reward=result.get('reward', 0.0),
                cost=result.get('cost', 0.0),
                is_solved=result.get('solved', False),
                llm_selection=result.get('llm_selection'),
                effective_reward=effective_rewards.get(role),
            )

        self._total_samples += 1

        # Track skip info if present
        skipped_roles = forward_result.get('skipped_roles', [])
        if skipped_roles:
            self.add_skip_info(skipped_roles)

    def add_skip_info(self, skipped_roles: List[str]) -> None:
        """
        Record skip information from one forward pass.

        Args:
            skipped_roles: List of roles that were skipped
        """
        self._skip_samples += 1
        for role in skipped_roles:
            self.skip_counts[role] = self.skip_counts.get(role, 0) + 1

    def get_skip_rates(self) -> Dict[str, float]:
        """
        Get skip rate for each role.

        Returns:
            Dictionary mapping role to skip rate (0.0 - 1.0)
        """
        if self._skip_samples == 0:
            return {}
        return {
            role: count / self._skip_samples
            for role, count in self.skip_counts.items()
        }

    def summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics.

        Returns:
            Dictionary with:
            - 'per_role': Dict[role, {solve_rate, mean_reward, mean_cost, ...}]
            - 'final': {solve_rate, mean_reward, total_cost, ...}
            - 'llm_distributions': Dict[role, {llm_name: count, ...}]
            - 'latest_llm_selections': Dict[role, List[str]] - LLMs selected in latest round
            - 'total_samples': int
        """
        per_role = {}
        llm_distributions = {}
        latest_llm_selections = {}

        for role, tracker in self.role_trackers.items():
            per_role[role] = tracker.summary()
            llm_distributions[role] = tracker.get_llm_distribution()
            latest_selection = tracker.get_latest_llm_selection()
            if latest_selection:
                latest_llm_selections[role] = latest_selection

        return {
            'per_role': per_role,
            'final': self.final_tracker.summary(),
            'llm_distributions': llm_distributions,
            'latest_llm_selections': latest_llm_selections,
            'total_samples': self._total_samples,
            'skip_rates': self.get_skip_rates(),
            'skip_samples': self._skip_samples,
        }

    def reset(self) -> None:
        """Clear all tracked data."""
        for tracker in self.role_trackers.values():
            tracker.reset()
        self.final_tracker.reset()
        self._total_samples = 0
        self.skip_counts.clear()
        self._skip_samples = 0

    def format_progress(self, iteration: int, total: int, loss: float, temp: float) -> str:
        """
        Format a progress string for logging.

        Args:
            iteration: Current iteration number
            total: Total iterations
            loss: Current loss value
            temp: Current temperature

        Returns:
            Formatted progress string
        """
        summary = self.summary()
        lines = [f"Iter {iteration}/{total} | Loss: {loss:.4f} | Temp: {temp:.2f}"]

        # Per-role stats
        for role, stats in summary['per_role'].items():
            solve_rate = stats.get('solve_rate', 0.0)
            mean_reward = stats.get('mean_reward', 0.0)
            mean_cost = stats.get('mean_cost', 0.0)
            lines.append(
                f"  {role:25s}: solve={solve_rate:.2f}, reward={mean_reward:.2f}, cost=${mean_cost:.3f}"
            )

            # LLM selection for this role - show latest and top 4 overall
            latest_selection = summary['latest_llm_selections'].get(role)
            if latest_selection:
                # Format as dict with worker1, worker2, ..., aggregator
                latest_dict = {}
                for i, llm in enumerate(latest_selection):
                    if i < len(latest_selection) - 1:
                        latest_dict[f"worker{i+1}"] = llm
                    else:
                        latest_dict["aggregator"] = llm
                lines.append(f"    Latest: {latest_dict}")

            llm_dist = summary['llm_distributions'].get(role, {})
            if llm_dist:
                top_llms = sorted(llm_dist.items(), key=lambda x: -x[1])[:4]
                llm_str = ", ".join(f"{llm}" for llm, _ in top_llms)
                lines.append(f"    Top-4:  [{llm_str}]")

        # Final stats
        final_stats = summary['final']
        lines.append(
            f"  {'Final':25s}: solve={final_stats.get('solve_rate', 0.0):.2f}, "
            f"reward={final_stats.get('mean_reward', 0.0):.2f}, "
            f"cost=${final_stats.get('mean_cost', 0.0):.3f}"
        )

        return "\n".join(lines)

    def format_final_summary(self) -> str:
        """
        Format a final summary string for end of training.

        Returns:
            Formatted summary string
        """
        summary = self.summary()
        lines = [
            "=" * 80,
            "TRAINING COMPLETE",
            "=" * 80,
            "",
            "Per-Role Statistics:",
            "-" * 80,
        ]

        # Header
        lines.append(f"{'Role':25s} | {'Solve Rate':10s} | {'Mean Reward':12s} | {'Mean Cost':10s}")
        lines.append("-" * 80)

        # Per-role stats
        for role, stats in summary['per_role'].items():
            lines.append(
                f"{role:25s} | {stats.get('solve_rate', 0.0):10.2f} | "
                f"{stats.get('mean_reward', 0.0):12.2f} | ${stats.get('mean_cost', 0.0):9.3f}"
            )

        # Final stats
        final_stats = summary['final']
        lines.append("-" * 80)
        lines.append(
            f"{'Final (Graph)':25s} | {final_stats.get('solve_rate', 0.0):10.2f} | "
            f"{final_stats.get('mean_reward', 0.0):12.2f} | ${final_stats.get('total_cost', 0.0):9.2f}"
        )

        # LLM distributions
        lines.extend(["", "LLM Selection Distributions:"])
        for role, llm_dist in summary['llm_distributions'].items():
            if llm_dist:
                total = sum(llm_dist.values())
                sorted_llms = sorted(llm_dist.items(), key=lambda x: -x[1])
                dist_str = ", ".join(f"{llm} ({count/total*100:.0f}%)" for llm, count in sorted_llms[:5])
                lines.append(f"  {role}: {dist_str}")

        lines.append("")
        lines.append(f"Total Samples: {summary['total_samples']}")
        lines.append(f"Total Cost: ${final_stats.get('total_cost', 0.0):.2f}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return self._total_samples

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'role_trackers': {
                role: {
                    'rewards': tracker.rewards.copy(),
                    'costs': tracker.costs.copy(),
                    'solved': tracker.solved.copy(),
                    'effective_rewards': tracker.effective_rewards.copy(),
                }
                for role, tracker in self.role_trackers.items()
            },
            'final_tracker': {
                'rewards': self.final_tracker.rewards.copy(),
                'costs': self.final_tracker.costs.copy(),
                'solved': self.final_tracker.solved.copy(),
            },
            'total_samples': self._total_samples,
            'skip_counts': self.skip_counts.copy(),
            'skip_samples': self._skip_samples,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        # Restore role trackers
        for role, tracker_state in state.get('role_trackers', {}).items():
            if role not in self.role_trackers:
                self.role_trackers[role] = RoleRewardTracker(role=role)
            tracker = self.role_trackers[role]
            tracker.rewards = tracker_state.get('rewards', [])
            tracker.costs = tracker_state.get('costs', [])
            tracker.solved = tracker_state.get('solved', [])
            tracker.effective_rewards = tracker_state.get('effective_rewards', [])

        # Restore final tracker
        final_state = state.get('final_tracker', {})
        self.final_tracker.rewards = final_state.get('rewards', [])
        self.final_tracker.costs = final_state.get('costs', [])
        self.final_tracker.solved = final_state.get('solved', [])

        # Restore total samples
        self._total_samples = state.get('total_samples', 0)

        # Restore skip data
        self.skip_counts = state.get('skip_counts', {})
        self._skip_samples = state.get('skip_samples', 0)
