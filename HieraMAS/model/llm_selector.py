"""
LLM Selector Neural Network for Learnable LLM Selection.

This module contains the neural network that learns to select optimal LLMs
for each position in a multi-agent pipeline using Gumbel-Softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class LLMSelector(nn.Module):
    """
    Neural network for selecting LLMs for each position in a multi-agent pipeline.

    Uses learnable position embeddings, task embeddings, and LLM embeddings as input,
    outputs selection probabilities using Gumbel-Softmax for differentiable sampling.

    Architecture:
        pos_indices -> pos_embedding -> pos_emb (N+1, 128)
        task_emb (1024) -> expand to (N+1, 1024)
        meta_emb (1024) -> expand to (N+1, 1024)
        concat(pos_emb, task_emb, meta_emb) -> row_proj -> row_h (N+1, hidden)
        llm_emb -> llm_proj -> llm_h (K+1, hidden)
        concat(row_h[i], llm_h[j]) -> score_net -> logits[i,j]
        logits -> Gumbel-Softmax -> selections
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        num_llms: int = 10,
        temperature: float = 2.0,
        max_positions: int = 20,
        pos_embed_dim: int = 128,
        random_llm: bool = False,
    ):
        """
        Initialize the LLM selector.

        Args:
            embed_dim: Dimension of LLM embeddings (1024 for gte-large-en-v1.5)
            hidden_dim: Dimension of hidden layers
            num_llms: Number of LLM options (including skip token)
            temperature: Initial Gumbel-Softmax temperature
            max_positions: Maximum number of positions (workers + aggregator)
            pos_embed_dim: Dimension of learnable position embeddings
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_llms = num_llms
        self.temperature = temperature
        self.max_positions = max_positions
        self.pos_embed_dim = pos_embed_dim
        self.random_llm = random_llm

        # Learnable position embeddings
        self.pos_embedding = nn.Embedding(max_positions, pos_embed_dim)

        # Row projection: concatenation of pos_emb + task_emb + meta_emb
        # Input: pos_embed_dim (128) + embed_dim (1024) + embed_dim (1024) = 2176
        row_input_dim = pos_embed_dim + embed_dim + embed_dim
        self.row_proj = nn.Sequential(
            nn.Linear(row_input_dim, hidden_dim)
        )

        # Project LLM embeddings (from sentence transformer dim)
        self.llm_proj = nn.Linear(embed_dim, hidden_dim)

        # Score network: combined features -> scalar score
        # Use MLP with non-linearity so row_h and llm_h can interact
        # (A single linear layer would make row contribution constant across LLMs,
        #  causing zero gradients for pos_embedding and row_proj due to softmax invariance)
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def compute_logits(
        self,
        num_positions: int,
        llm_emb: torch.Tensor,
        task_emb: Optional[torch.Tensor] = None,
        meta_emb: Optional[torch.Tensor] = None,
        aggregator_mask: bool = True,
        is_training: bool = True,
        excluded_llm_indices: Optional[List[int]] = None,
        allow_aggregator_skip: bool = False,
    ) -> torch.Tensor:
        """
        Compute logits for all (position, LLM) pairs.

        Args:
            num_positions: Number of positions (N+1 = workers + aggregator)
            llm_emb: LLM embeddings (K+1, embed_dim)
            task_emb: Task sentence embedding (embed_dim,), optional
            meta_emb: Task metadata embedding (embed_dim,), optional
            aggregator_mask: If True, mask out skip token for aggregator
            is_training: If True, clamp logits to [-10, 10]; if False, no clamp
            excluded_llm_indices: List of LLM indices to exclude from selection
            allow_aggregator_skip: If True, allow aggregator to select skip token
                                   (overrides aggregator_mask for skip token only)

        Returns:
            Logits tensor (N+1, K+1)
        """
        N_plus_1 = num_positions
        K_plus_1 = llm_emb.shape[0]
        device = llm_emb.device

        # Generate learnable position embeddings
        pos_indices = torch.arange(N_plus_1, device=device)
        pos_emb = self.pos_embedding(pos_indices)  # (N+1, pos_embed_dim)

        # Handle task/meta embeddings with fallback to zeros for backward compatibility
        if task_emb is None:
            task_emb = torch.zeros(self.embed_dim, device=device)
        if meta_emb is None:
            meta_emb = torch.zeros(self.embed_dim, device=device)

        # Expand task/meta embeddings to all positions (they are shared across positions)
        task_emb_exp = task_emb.unsqueeze(0).expand(N_plus_1, -1)  # (N+1, embed_dim)
        meta_emb_exp = meta_emb.unsqueeze(0).expand(N_plus_1, -1)  # (N+1, embed_dim)

        # Concatenate all three components: pos_emb + task_emb + meta_emb
        row_input = torch.cat([pos_emb, task_emb_exp, meta_emb_exp], dim=-1)
        # Shape: (N+1, pos_embed_dim + embed_dim + embed_dim) = (N+1, 2176)

        # Project row input through the sequential network
        row_h = self.row_proj(row_input)  # (N+1, hidden)
        llm_h = self.llm_proj(llm_emb)  # (K+1, hidden)

        # Compute logits efficiently using broadcasting
        # Expand for pairwise combination
        row_h_exp = row_h.unsqueeze(1).expand(-1, K_plus_1, -1)  # (N+1, K+1, hidden)
        llm_h_exp = llm_h.unsqueeze(0).expand(N_plus_1, -1, -1)  # (N+1, K+1, hidden)

        # Concatenate and compute scores
        combined = torch.cat([row_h_exp, llm_h_exp], dim=-1)  # (N+1, K+1, 2*hidden)
        logits = self.score_net(combined).squeeze(-1)  # (N+1, K+1)

        # If random_llm mode, zero out all logits for uniform random selection
        if self.random_llm:
            print(f"[LLMSelector] random_llm=True, zeroing out logits. Original logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            logits = torch.zeros_like(logits)
            print(f"[LLMSelector] After zeroing: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")

        # Clip logits to prevent "logit explosion" that causes gradient vanishing
        # When logits become too large (e.g., 300+), softmax becomes one-hot,
        # log(1.0) = 0, and gradients vanish. Only clamp during training.
        if is_training:
            logits = torch.clamp(logits, min=-10.0, max=10.0)

        # Mask skip token for aggregator (last position, last LLM option)
        # Note: mask is applied AFTER clipping so the -1e10 value is preserved
        # If allow_aggregator_skip is True, skip token is NOT masked (for skip learning)
        if aggregator_mask and not allow_aggregator_skip:
            # Create mask in a way that doesn't interfere with clipping
            logits = logits.clone()  # Ensure we don't modify in-place
            logits[-1, -1] = -1e10  # Large negative value to mask skip token

        # Mask excluded LLM indices (e.g., for restricting to OpenRouter models only)
        if excluded_llm_indices is not None:
            if not aggregator_mask:
                logits = logits.clone()  # Ensure we don't modify in-place
            for idx in excluded_llm_indices:
                logits[:, idx] = -1e8  # Large negative value to exclude these LLMs

        return logits

    def forward(
        self,
        num_positions: int,
        llm_emb: torch.Tensor,
        task_emb: Optional[torch.Tensor] = None,
        meta_emb: Optional[torch.Tensor] = None,
        hard: bool = False,
        aggregator_mask: bool = True,
        is_training: bool = True,
        allow_aggregator_skip: bool = False,
        excluded_llm_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: select LLMs for each position.

        Args:
            num_positions: Number of positions (N+1 = workers + aggregator)
            llm_emb: LLM embeddings (K+1, embed_dim)
            task_emb: Task sentence embedding (embed_dim,), optional
            meta_emb: Task metadata embedding (embed_dim,), optional
            hard: If True, use argmax; else use Gumbel-Softmax
            aggregator_mask: If True, mask out skip token for aggregator
            is_training: If True, clamp logits and use self.temperature;
                        if False, no clamp and use fixed T=10
            allow_aggregator_skip: If True, allow aggregator to select skip token
                                   (for skip learning)
            excluded_llm_indices: List of LLM indices to exclude from selection

        Returns:
            selections: Selected LLM indices (N+1,)
            log_probs: Log probabilities of selections (N+1,)
            entropy: Entropy of the selection distribution (scalar)
            skip_probs: Skip token probabilities for each position (N+1,)
        """
        # Compute logits
        logits = self.compute_logits(
            num_positions, llm_emb,
            task_emb=task_emb,
            meta_emb=meta_emb,
            aggregator_mask=aggregator_mask,
            is_training=is_training,
            allow_aggregator_skip=allow_aggregator_skip,
            excluded_llm_indices=excluded_llm_indices,
        )  # (N+1, K+1)

        # Temperature-scaled logits for stable probabilities
        # Use fixed T=0.7 during inference for sharper distributions
        if is_training:
            scaled_logits = logits / self.temperature
        else:
            scaled_logits = logits / 0.7

        # Compute probabilities and log probabilities
        probs = F.softmax(scaled_logits, dim=-1)  # (N+1, K+1)
        log_probs_all = F.log_softmax(scaled_logits, dim=-1)  # (N+1, K+1)

        # Compute entropy: H = -sum(p * log(p))
        # Entropy regularization prevents policy collapse
        entropy = -(probs * log_probs_all).sum()

        # Extract skip probabilities (last column is skip token)
        skip_probs = probs[:, -1]  # (N+1,)

        if hard:
            # Argmax selection (for evaluation)
            selections = logits.argmax(dim=-1)  # (N+1,)
            # Get log prob of selected actions (non-differentiable for eval)
            log_probs = log_probs_all.gather(1, selections.unsqueeze(1)).squeeze(1)  # (N+1,)
        else:
            # Gumbel-Softmax for differentiable sampling
            # hard=True gives one-hot but with gradients via straight-through estimator
            gumbel_out = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            selections = gumbel_out.argmax(dim=-1)  # (N+1,) - for returning discrete indices
            # Use gather to avoid -inf * 0 = NaN issue
            # Gradients flow through log_probs_all to update network parameters
            log_probs = log_probs_all.gather(1, selections.unsqueeze(1)).squeeze(1)  # (N+1,)

        return selections, log_probs, entropy, skip_probs

    def set_temperature(self, temperature: float):
        """Set the Gumbel-Softmax temperature."""
        self.temperature = max(temperature, 0.01)  # Prevent division by zero

    def get_selection_probs(
        self,
        num_positions: int,
        llm_emb: torch.Tensor,
        task_emb: Optional[torch.Tensor] = None,
        meta_emb: Optional[torch.Tensor] = None,
        aggregator_mask: bool = True,
        is_training: bool = True,
        allow_aggregator_skip: bool = False,
    ) -> torch.Tensor:
        """
        Get selection probabilities without sampling.

        Useful for visualization and analysis.

        Args:
            num_positions: Number of positions (N+1 = workers + aggregator)
            llm_emb: LLM embeddings (K+1, embed_dim)
            task_emb: Task sentence embedding (embed_dim,), optional
            meta_emb: Task metadata embedding (embed_dim,), optional
            aggregator_mask: If True, mask out skip token for aggregator
            is_training: If True, clamp logits; if False, no clamp
            allow_aggregator_skip: If True, allow aggregator to select skip token
                                   (overrides aggregator_mask for skip token only)

        Returns:
            Probabilities tensor (N+1, K+1)
        """
        logits = self.compute_logits(
            num_positions, llm_emb,
            task_emb=task_emb,
            meta_emb=meta_emb,
            aggregator_mask=aggregator_mask,
            is_training=is_training,
            allow_aggregator_skip=allow_aggregator_skip,
        )
        return F.softmax(logits, dim=-1)


class TemperatureScheduler:
    """
    Temperature annealing scheduler for Gumbel-Softmax.

    Supports exponential, linear, and cosine decay schedules.
    Higher temperature = more exploration, lower = more exploitation.
    """

    def __init__(
        self,
        initial_temp: float = 10.0,
        min_temp: float = 0.1,
        total_steps: int = 100,
        schedule: str = 'cosine',
    ):
        """
        Initialize the temperature scheduler.

        Args:
            initial_temp: Starting temperature
            min_temp: Minimum temperature (floor)
            total_steps: Total number of steps for annealing
            schedule: One of 'linear', 'cosine', 'exponential'
        """
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.total_steps = total_steps
        self.schedule = schedule
        self.step_count = 0

        # Auto-calculate decay_rate for exponential schedule
        if schedule == 'exponential':
            self.decay_rate = (min_temp / initial_temp) ** (1.0 / total_steps)
        else:
            self.decay_rate = None

    def step(self):
        """Increment step counter."""
        self.step_count += 1

    def get_temperature(self) -> float:
        """
        Get current temperature.

        Returns:
            Current temperature value
        """
        progress = min(self.step_count / self.total_steps, 1.0)

        if self.schedule == 'linear':
            temp = self.initial_temp - progress * (self.initial_temp - self.min_temp)

        elif self.schedule == 'cosine':
            import math
            temp = self.min_temp + 0.5 * (self.initial_temp - self.min_temp) * (
                1 + math.cos(math.pi * progress)
            )

        elif self.schedule == 'exponential':
            temp = self.initial_temp * (self.decay_rate ** self.step_count)

        else:
            temp = self.initial_temp

        return max(temp, self.min_temp)

    def reset(self):
        """Reset step counter."""
        self.step_count = 0

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            'step_count': self.step_count,
            'initial_temp': self.initial_temp,
            'min_temp': self.min_temp,
            'schedule': self.schedule,
            'total_steps': self.total_steps,
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.step_count = state.get('step_count', 0)
        self.initial_temp = state.get('initial_temp', self.initial_temp)
        self.min_temp = state.get('min_temp', self.min_temp)
        self.schedule = state.get('schedule', self.schedule)
        self.total_steps = state.get('total_steps', self.total_steps)
        # Recalculate decay_rate if exponential
        if self.schedule == 'exponential':
            self.decay_rate = (self.min_temp / self.initial_temp) ** (1.0 / self.total_steps)
        else:
            self.decay_rate = None
