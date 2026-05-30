"""
Embedding Generator for Learnable LLM Selection.

This module generates embeddings for LLM profiles and position descriptions
using the sentence transformer model.
"""

import hashlib
import torch
from typing import Dict, Any, Optional, List

from HieraMAS.llm.profile_embedding import get_sentence_embedding
from HieraMAS.prompt.llm_profiles import LLM_PROFILES, LLM_OPTIONS
from HieraMAS.prompt.position_prompts import PositionPromptGenerator, create_default_task_info


class PositionLLMEmbedder:
    """
    Generates embeddings for positions and LLMs using sentence transformers.

    LLM embeddings are cached since they are static.
    Position embeddings are computed per-task since they depend on task info.
    """

    def __init__(self, embed_dim: int = 1024):
        """
        Initialize the embedder.

        Args:
            embed_dim: Expected embedding dimension (1024 for gte-large-en-v1.5)
        """
        self.embed_dim = embed_dim
        self._llm_embeddings_cache: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        # Task and metadata embedding caches (in-memory, per session)
        self._task_emb_cache: Dict[str, torch.Tensor] = {}
        self._meta_emb_cache: Dict[str, torch.Tensor] = {}
        self._task_role_emb_cache: Dict[str, torch.Tensor] = {}

    def get_llm_embeddings(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get embeddings for all LLM options.

        Cached since LLM profiles are static and don't change between tasks.

        Args:
            device: Target device for tensor

        Returns:
            Tensor of shape (K+1, embed_dim) where K+1 is len(LLM_OPTIONS)
        """
        if self._llm_embeddings_cache is not None:
            if device is not None and self._llm_embeddings_cache.device != device:
                self._llm_embeddings_cache = self._llm_embeddings_cache.to(device)
            return self._llm_embeddings_cache

        embeddings = []
        for llm_name in LLM_OPTIONS:
            profile = LLM_PROFILES[llm_name]
            emb = get_sentence_embedding(profile)
            embeddings.append(torch.tensor(emb, dtype=torch.float32))

        self._llm_embeddings_cache = torch.stack(embeddings)  # (K+1, embed_dim)

        if device is not None:
            self._llm_embeddings_cache = self._llm_embeddings_cache.to(device)
            self._device = device

        return self._llm_embeddings_cache

    def get_position_embeddings(
        self,
        num_workers: int,
        task_info: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        role: Optional[str] = None,
        role_description: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Get embeddings for all positions (workers + aggregator).

        Not cached since these depend on task-specific information.

        Args:
            num_workers: Number of worker agents
            task_info: Task metadata dict (type, difficulty, domain, etc.)
            device: Target device for tensor
            role: Optional role name (e.g., "Math Solver")
            role_description: Optional detailed description of the role

        Returns:
            Tensor of shape (N+1, embed_dim) where N+1 = num_workers + 1 (aggregator)
        """
        if task_info is None:
            task_info = create_default_task_info()

        # Generate position-specific prompts with role info
        prompts = PositionPromptGenerator.get_all_position_prompts(
            num_workers, task_info, role, role_description
        )

        embeddings = []
        for prompt in prompts:
            emb = get_sentence_embedding(prompt)
            embeddings.append(torch.tensor(emb, dtype=torch.float32))

        result = torch.stack(embeddings)  # (N+1, embed_dim)

        if device is not None:
            result = result.to(device)

        return result

    def clear_cache(self):
        """Clear the cached LLM embeddings."""
        self._llm_embeddings_cache = None

    def clear_task_caches(self):
        """Clear task and metadata embedding caches."""
        self._task_emb_cache.clear()
        self._meta_emb_cache.clear()
        self._task_role_emb_cache.clear()

    def get_task_embedding(
        self,
        task: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get sentence embedding for the task text.

        Results are cached in memory to avoid redundant embedding computation.

        Args:
            task: The task string
            device: Target device for tensor

        Returns:
            Task embedding tensor of shape (embed_dim,)
        """
        # Create cache key from task hash (truncate to 512 tokens worth of chars)
        task_truncated = task[:2000]  # Approximate 512 tokens
        cache_key = hashlib.md5(task_truncated.encode()).hexdigest()

        if cache_key not in self._task_emb_cache:
            emb = get_sentence_embedding(task_truncated)
            self._task_emb_cache[cache_key] = torch.tensor(emb, dtype=torch.float32)

        result = self._task_emb_cache[cache_key]
        if device is not None:
            result = result.to(device)
        return result

    def get_meta_embedding(
        self,
        metadata: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get sentence embedding for task metadata.

        Formats metadata dict into a descriptive string, then embeds it.
        Results are cached based on metadata content.

        Args:
            metadata: Task metadata dict (type, difficulty, domain, key_skills, etc.)
            device: Target device for tensor

        Returns:
            Metadata embedding tensor of shape (embed_dim,)
        """
        # Format metadata into a descriptive string
        meta_str = self._format_metadata(metadata)

        # Create cache key
        cache_key = hashlib.md5(meta_str.encode()).hexdigest()

        if cache_key not in self._meta_emb_cache:
            emb = get_sentence_embedding(meta_str)
            self._meta_emb_cache[cache_key] = torch.tensor(emb, dtype=torch.float32)

        result = self._meta_emb_cache[cache_key]
        if device is not None:
            result = result.to(device)
        return result

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata dictionary into a descriptive string for embedding.

        Example output:
        "Task type: math. Difficulty: medium. Domain: algebra.
         Key skills: equation solving, arithmetic calculation."

        Args:
            metadata: Task metadata dict

        Returns:
            Formatted string describing the task metadata
        """
        parts = []

        if 'type' in metadata:
            parts.append(f"Task type: {metadata['type']}")
        if 'difficulty' in metadata:
            parts.append(f"Difficulty: {metadata['difficulty']}")
        if 'domain' in metadata:
            parts.append(f"Domain: {metadata['domain']}")
        if 'key_skills' in metadata and metadata['key_skills']:
            skills = metadata['key_skills']
            if isinstance(skills, list):
                skills = ', '.join(skills)
            parts.append(f"Key skills: {skills}")
        if 'description' in metadata and metadata['description']:
            parts.append(f"Description: {metadata['description']}")

        return '. '.join(parts) + '.' if parts else "General task."

    def get_task_role_embedding(
        self,
        task: str,
        role: Optional[str] = None,
        role_description: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get sentence embedding for task text merged with role information.

        This method merges the task text with role description to create a role-aware
        task embedding. For backward compatibility, if role is None, it returns just
        the task embedding (same as old behavior).

        Args:
            task: The task string
            role: Role name (e.g., "Math Solver"), optional for backward compatibility
            role_description: Detailed role description, optional for backward compatibility
            device: Target device for tensor

        Returns:
            Task+role embedding tensor of shape (embed_dim,), or task-only if role is None
        """
        # Truncate task to avoid token limits (approximate 512 tokens)
        task_truncated = task[:1500]

        # Build combined text based on whether role is provided
        if role is not None and role_description is not None:
            # New behavior: merge task + role
            combined_text = f"Task: {task_truncated}\n\nRole: {role}\n{role_description}"
        else:
            # Backward compatibility: just task (same as old get_task_embedding)
            combined_text = f"Task: {task_truncated}"

        # Create cache key from combined text hash
        cache_key = hashlib.md5(combined_text.encode()).hexdigest()

        if cache_key not in self._task_role_emb_cache:
            emb = get_sentence_embedding(combined_text)
            self._task_role_emb_cache[cache_key] = torch.tensor(emb, dtype=torch.float32)

        result = self._task_role_emb_cache[cache_key]
        if device is not None:
            result = result.to(device)
        return result

    @property
    def num_llms(self) -> int:
        """Return the number of LLM options."""
        return len(LLM_OPTIONS)


def get_combined_embedding(
    pos_idx: int,
    llm_idx: int,
    pos_embeddings: torch.Tensor,
    llm_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Get a combined embedding for a (position, LLM) pair.

    This can be used for computing compatibility scores.

    Args:
        pos_idx: Position index
        llm_idx: LLM index
        pos_embeddings: Position embeddings tensor (N+1, D)
        llm_embeddings: LLM embeddings tensor (K+1, D)

    Returns:
        Combined embedding tensor (2*D,)
    """
    return torch.cat([pos_embeddings[pos_idx], llm_embeddings[llm_idx]], dim=-1)
