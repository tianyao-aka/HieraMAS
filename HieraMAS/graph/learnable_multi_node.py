"""
Learnable MultiNode with LLM Selection.

This module combines a learnable LLM selector with the existing MultiNode
execution logic. The selector learns to choose optimal LLMs for each
worker and aggregator position based on task characteristics.
"""

import random
import torch
import torch.nn as nn
from typing import List, Any, Dict, Tuple, Optional

from HieraMAS.model.llm_selector import LLMSelector
from HieraMAS.model.embeddings import PositionLLMEmbedder
from HieraMAS.utils.task_cache import TaskMetaCache
from HieraMAS.prompt.llm_profiles import LLM_OPTIONS
from HieraMAS.prompt.role_descriptions import get_role_description
from HieraMAS.graph.multi_node import MultiNode
from HieraMAS.utils.globals import Cost, PromptTokens, CompletionTokens


class LearnableMultiNode(nn.Module):
    """
    Combines learnable LLM selection with MultiNode execution.

    Flow:
    1. LLMSelector samples LLM assignments (returns log_probs for training)
    2. Create MultiNode with selected worker_llm_names and aggregator_llm_name
    3. MultiNode executes as usual
    4. Return (output, log_probs, cost, llm_selection) for policy gradient training

    The only trainable parameters are in the LLMSelector.

    Enhanced for graph usage:
    - Accepts spatial_info/temporal_info from predecessor nodes
    - Builds enhanced task prompt with predecessor outputs
    - Returns selected LLM names for logging
    """

    def __init__(
        self,
        num_workers: int = 3,
        agent_name: str = "MathSolver",
        aggregator_method: str = "FinalMultiAgentAggregator",
        domain: str = "gsm8k",
        role: str = "Math Solver",
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        initial_temperature: float = 2.0,
        use_llm_for_task_meta: bool = True,
        node_id: Optional[str] = None,
        external_selector: Optional[LLMSelector] = None,
        use_qwen3_80b: bool = False,
        random_llm: bool = False,
        allow_aggregator_skip: bool = False,
        excluded_llm_indices: Optional[List[int]] = None,
        fixed_llm: bool = False,
        fixed_llm_name: Optional[str] = None,
        aggregator_only: bool = False,
    ):
        """
        Initialize the learnable multi-node.

        Args:
            num_workers: Number of worker positions
            agent_name: Agent type for workers (e.g., "MathSolver")
            aggregator_method: Aggregator agent type
            domain: Domain name (e.g., "gsm8k")
            role: Role for all workers
            embed_dim: Embedding dimension (1024 for gte-large-en-v1.5)
            hidden_dim: Hidden dimension for selector network
            initial_temperature: Initial Gumbel-Softmax temperature
            use_llm_for_task_meta: Whether to use LLM for task metadata extraction
            node_id: Optional unique identifier for this node
            external_selector: Optional external LLMSelector to share across nodes
            use_qwen3_80b: Restrict LLM selection to OpenRouter models only (exclude GPT and DeepSeek-v3)
            allow_aggregator_skip: If True, allow aggregator to select skip token (for skip learning)
            excluded_llm_indices: List of LLM indices to exclude from selection (e.g., [2] for deepseek-r1-14b)
            fixed_llm: If True, use fixed_llm_name for all workers/aggregators (no LLM selection)
            fixed_llm_name: LLM name to use when fixed_llm=True
            aggregator_only: If True, skip workers and use aggregator directly (baseline mode)
        """
        super().__init__()

        self.num_workers = num_workers
        self.aggregator_only = aggregator_only
        self.agent_name = agent_name
        self.aggregator_method = aggregator_method
        self.domain = domain
        self.role = role
        self.use_llm_for_task_meta = use_llm_for_task_meta
        self.use_qwen3_80b = use_qwen3_80b
        self.random_llm = random_llm
        self.allow_aggregator_skip = allow_aggregator_skip
        self.excluded_llm_indices = excluded_llm_indices
        self.fixed_llm = fixed_llm
        self.fixed_llm_name = fixed_llm_name
        self.node_id = node_id or f"learnable_mn_{role.replace(' ', '_')}"

        # Get role description for position embeddings
        self.role_description = get_role_description(role)

        # Use external selector if provided, otherwise create own
        if external_selector is not None:
            self.selector = external_selector
            self._owns_selector = False
        else:
            self.selector = LLMSelector(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_llms=len(LLM_OPTIONS),
                temperature=initial_temperature,
                random_llm=self.random_llm,
            )
            self._owns_selector = True

        # Embedder (uses cached LLM embeddings)
        self.embedder = PositionLLMEmbedder(embed_dim=embed_dim)

        # Task metadata cache
        self.task_cache = TaskMetaCache()

        # Tracking attributes for last execution
        self._last_llm_selection: List[str] = []
        self._last_output: str = ""
        self._last_cost: float = 0.0
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0

    def select_llms(
        self,
        task: str,
        hard: bool = False,
        is_training: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select LLMs for each position based on task.

        Now incorporates:
        - Task sentence embedding (semantic content of the task)
        - Task metadata embedding (structured info: type, difficulty, domain, skills)

        Args:
            task: Task string
            hard: If True, use argmax; else use Gumbel-Softmax
            is_training: If True, clamp logits; if False, no clamp and use T=10

        Returns:
            llm_names: List of LLM names for [worker0, worker1, ..., aggregator]
            log_probs: Log probabilities for policy gradient (N+1,)
            entropy: Entropy of the selection distribution (scalar)
            skip_probs: Skip token probabilities for each position (N+1,)
        """
        # Get device
        device = next(self.selector.parameters()).device

        # 1. Get LLM profile embeddings (cached, static)
        llm_emb = self.embedder.get_llm_embeddings(device=device)

        # 2. Get task metadata via cache (uses gpt-4o-mini if not cached)
        metadata = self.task_cache.get_or_extract(task, use_llm=self.use_llm_for_task_meta)

        # 3. Get merged task+role embedding (NEW - backward compatible)
        task_role_emb = self.embedder.get_task_role_embedding(
            task,
            role=self.role,
            role_description=self.role_description,
            device=device
        )

        # 4. Get metadata embedding (unchanged)
        meta_emb = self.embedder.get_meta_embedding(metadata, device=device)

        # Number of positions = workers + aggregator
        num_positions = self.num_workers + 1

        # Run selector with task+role merged embedding
        selections, log_probs, entropy, skip_probs = self.selector(
            num_positions,
            llm_emb,
            task_emb=task_role_emb,  # Now contains task + role
            meta_emb=meta_emb,        # Unchanged
            hard=hard,
            aggregator_mask=True,
            is_training=is_training,
            allow_aggregator_skip=self.allow_aggregator_skip,
            excluded_llm_indices=self.excluded_llm_indices,
        )

        # Convert indices to LLM names
        llm_names = [LLM_OPTIONS[idx] for idx in selections.tolist()]

        return llm_names, log_probs, entropy, skip_probs

    def select_llms_topN_rotation(
        self,
        task: str,
        top_n: int = 2,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select LLMs using top-N rotation strategy for inference.

        Strategy:
        1. Randomly select 1 worker position
        2. Get logits for that position, take top-N LLMs
        3. Distribute workers in rotation: [A, B, C, A, B, C, ...] for N=3
        4. Aggregator uses argmax (top-1)

        Args:
            task: Task string
            top_n: Number of top LLMs to rotate (default=2)

        Returns:
            llm_names: List of LLM names [worker0, worker1, ..., aggregator]
            log_probs: Zero tensor (no gradient needed for eval)
            entropy: Zero tensor (deterministic selection)
            skip_probs: Zero tensor (not used for inference)
        """
        device = next(self.selector.parameters()).device

        # 1. Get LLM profile embeddings (cached, static)
        llm_emb = self.embedder.get_llm_embeddings(device=device)

        # 2. Get task metadata
        metadata = self.task_cache.get_or_extract(task, use_llm=self.use_llm_for_task_meta)

        # 3. Get merged task+role embedding
        task_role_emb = self.embedder.get_task_role_embedding(
            task,
            role=self.role,
            role_description=self.role_description,
            device=device
        )

        # 4. Get metadata embedding
        meta_emb = self.embedder.get_meta_embedding(metadata, device=device)

        num_positions = self.num_workers + 1

        # 5. Get raw logits (not sampled) - use is_training=False for no clamping
        logits = self.selector.compute_logits(
            num_positions,
            llm_emb,
            task_emb=task_role_emb,
            meta_emb=meta_emb,
            aggregator_mask=True,
            is_training=False,
            excluded_llm_indices=self.excluded_llm_indices,
        )  # Shape: (num_workers+1, num_llms+1)

        # 6. Randomly select one worker position for top-N selection
        random_worker_idx = random.randint(0, self.num_workers - 1)
        worker_logits = logits[random_worker_idx, :-1]  # Exclude skip token for workers

        # 7. Get top-N LLM indices from selected position
        topN_indices = torch.topk(worker_logits, k=top_n).indices
        topN_llms = [LLM_OPTIONS[idx.item()] for idx in topN_indices]

        # 8. Distribute workers in rotation pattern [A, B, C, A, B, C, ...]
        worker_llms = [topN_llms[i % top_n] for i in range(self.num_workers)]

        # 9. Aggregator uses argmax (top-1)
        aggregator_logits = logits[-1, :]  # Last position (aggregator)
        aggregator_idx = aggregator_logits.argmax().item()
        aggregator_llm = LLM_OPTIONS[aggregator_idx]

        # 10. Combine all selections
        llm_names = worker_llms + [aggregator_llm]

        # 11. Return dummy log_probs, entropy, and skip_probs (no gradients needed for eval)
        log_probs = torch.zeros(num_positions, device=device)
        entropy = torch.tensor(0.0, device=device)
        skip_probs = torch.zeros(num_positions, device=device)

        return llm_names, log_probs, entropy, skip_probs

    def create_multi_node(
        self,
        worker_llm_names: List[str],
        aggregator_llm_name: str,
    ) -> MultiNode:
        """
        Create a MultiNode with the selected LLMs.

        Args:
            worker_llm_names: LLM names for workers (may include 'skip')
            aggregator_llm_name: LLM name for aggregator

        Returns:
            Configured MultiNode instance
        """
        # Filter out skipped workers
        active_worker_llms = [llm for llm in worker_llm_names if llm != 'skip']

        # Handle edge case: all workers skipped
        if not active_worker_llms:
            active_worker_llms = ['gpt-4o-mini']  # Fallback to at least one worker

        return MultiNode(
            id=self.node_id,
            agent_name=self.agent_name,
            num_agents=len(active_worker_llms),
            aggregator_method=self.aggregator_method,
            domain=self.domain,
            llm_name=active_worker_llms[0],  # Default fallback
            role=self.role,
            worker_llm_names=active_worker_llms,
            aggregator_llm_name=aggregator_llm_name,
        )

    def _build_enhanced_task(
        self,
        original_task: str,
        spatial_info: Optional[Dict[str, Dict]] = None,
        temporal_info: Optional[Dict[str, Dict]] = None,
    ) -> str:
        """
        Build an enhanced task prompt that includes predecessor outputs.

        Args:
            original_task: The original task string
            spatial_info: Dict mapping node_id to {'role': str, 'output': str}
                         from predecessor nodes in the current round
            temporal_info: Dict mapping node_id to {'role': str, 'output': str}
                          from previous round outputs

        Returns:
            Enhanced task string with predecessor context

        Example output:
            "The task is: Josh decides to flip a house...

            At the same time, the outputs of other agents are as follows:

            Agent mn_Math_xxxx, role is Mathematical Analyst, output is:
            Let x = purchase price = 80000...
            The answer is 70000"
        """
        enhanced_parts = [original_task]

        # Add spatial info (from predecessor nodes in same round)
        if spatial_info:
            spatial_outputs = []
            for node_id, info in spatial_info.items():
                role = info.get('role', 'Unknown')
                output = info.get('output', '')
                if output:
                    spatial_outputs.append(
                        f"Agent {node_id}, role is {role}, output is:\n{output}"
                    )

            if spatial_outputs:
                enhanced_parts.append(
                    "\nAt the same time, the outputs of other agents are as follows:\n\n"
                    + "\n\n".join(spatial_outputs)
                )

        # Add temporal info (from previous round)
        if temporal_info:
            temporal_outputs = []
            for node_id, info in temporal_info.items():
                role = info.get('role', 'Unknown')
                output = info.get('output', '')
                if output:
                    temporal_outputs.append(
                        f"Agent {node_id}, role is {role}, previous output is:\n{output}"
                    )

            if temporal_outputs:
                enhanced_parts.append(
                    "\nFrom the previous round, the outputs are:\n\n"
                    + "\n\n".join(temporal_outputs)
                )

        return "\n".join(enhanced_parts)

    async def _forward_aggregator_only(
        self,
        original_input: Dict[str, Any],
        enhanced_input: Dict[str, Any],
        hard: bool,
        is_training: bool,
    ) -> Dict[str, Any]:
        """
        Execute aggregator-only mode (baseline): skip workers, call aggregator directly.

        In this mode:
        - Workers are skipped entirely
        - Aggregator receives the task directly and answers without worker synthesis
        - Uses baseline prompt (direct answering, no worker outputs to combine)

        Args:
            original_input: Original input dict with 'task' key
            enhanced_input: Input with predecessor info added
            hard: If True, use argmax selection
            is_training: Training mode flag

        Returns:
            Dict with same format as forward(), but with aggregator-only results
        """
        from HieraMAS.prompt.prompt_set_registry import PromptSetRegistry
        from HieraMAS.llm.llm_registry import LLMRegistry

        task = enhanced_input.get('task', '')

        # Get aggregator LLM name
        if self.fixed_llm:
            aggregator_llm = self.fixed_llm_name
        else:
            # Use first LLM in options as default for baseline
            aggregator_llm = LLM_OPTIONS[0]

        # Get baseline prompt for this domain and role
        prompt_set = PromptSetRegistry.get(self.domain)

        # Build baseline prompt (direct answering without worker outputs)
        if hasattr(prompt_set, 'get_baseline_prompt'):
            system_prompt = prompt_set.get_baseline_prompt(self.role)
        else:
            # Fallback: use role description + constraint
            role_desc = prompt_set.get_description(self.role) if hasattr(prompt_set, 'get_description') else ""
            if self.domain == 'mmlu':
                system_prompt = f"""{role_desc}
Analyze this multiple choice question and provide your answer.
The first line must contain only A, B, C, or D.
Reply in less than 100 words."""
            elif self.domain in ('gsm8k', 'math'):
                system_prompt = f"""{role_desc}
Solve this math problem step by step.
End with "The answer is [number]"."""
            elif self.domain == 'humaneval':
                system_prompt = f"""{role_desc}
Implement the following function.
Reply with only Python code in a code block."""
            else:
                system_prompt = role_desc

        # Build messages for LLM call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        # Track cost and tokens
        cost_before = Cost.instance().value
        prompt_before = PromptTokens.instance().value
        completion_before = CompletionTokens.instance().value

        # Get LLM and make call
        llm = LLMRegistry.get(aggregator_llm)
        output = await llm.agen(messages)

        cost_after = Cost.instance().value
        cost = cost_after - cost_before

        # Compute token counts
        prompt_tokens = int(PromptTokens.instance().value - prompt_before)
        completion_tokens = int(CompletionTokens.instance().value - completion_before)

        # Update tracking attributes
        self._last_llm_selection = [aggregator_llm]  # Only aggregator
        self._last_output = output if isinstance(output, str) else str(output)
        self._last_cost = cost
        self._last_prompt_tokens = prompt_tokens
        self._last_completion_tokens = completion_tokens

        # Return dummy log_probs and entropy (no training in baseline mode)
        device = next(self.selector.parameters()).device
        log_probs = torch.zeros(1, device=device)  # Single position (aggregator only)
        entropy = torch.tensor(0.0, device=device)
        skip_probs = torch.zeros(1, device=device)

        return {
            'output': output,
            'log_probs': log_probs,
            'cost': cost,
            'llm_selection': [aggregator_llm],
            'entropy': entropy,
            'skip_probs': skip_probs,
            'aggregator_skip_prob': torch.tensor(0.0, device=device),
            'is_skipped': False,
            'aggregator_only': True,  # Mark as aggregator-only mode
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
        }

    async def forward(
        self,
        input: Dict[str, Any],
        spatial_info: Optional[Dict[str, Dict]] = None,
        temporal_info: Optional[Dict[str, Dict]] = None,
        hard: bool = False,
        is_training: bool = True,
        inference_strategy: str = "standard",
        rotation_top_n: int = 2,
        force_skip: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute with learned LLM selection.

        Args:
            input: Input dict with 'task' key
            spatial_info: Dict mapping node_id to {'role': str, 'output': str}
                         from predecessor nodes in the current round
            temporal_info: Dict mapping node_id to {'role': str, 'output': str}
                          from previous round outputs
            hard: If True, use argmax selection; else use Gumbel-Softmax
            is_training: If True, clamp logits; if False, no clamp and use T=10
            inference_strategy: LLM selection strategy ("standard" or "topN_rotation")
            rotation_top_n: Number of top LLMs to rotate for topN_rotation strategy
            force_skip: If True, skip this node without LLM calls (for role skip inference)

        Returns:
            Dict containing:
                output: The aggregator's final output
                log_probs: Log probabilities of LLM selections (for policy gradient)
                cost: Actual API cost incurred
                llm_selection: List of selected LLM names [workers..., aggregator]
                entropy: Entropy of the LLM selection distribution (for regularization)
                skip_probs: Skip token probabilities for each position (N+1,)
                aggregator_skip_prob: Skip probability for aggregator position (scalar)
                is_skipped: Whether this node was skipped (aggregator selected skip)
        """
        # Check if forced skip (for role skip inference)
        if force_skip:
            # Return skip result without LLM calls
            return {
                'output': '',
                'log_probs': torch.zeros(self.num_workers + 1),
                'cost': 0.0,
                'llm_selection': ['skip'] * (self.num_workers + 1),
                'entropy': torch.tensor(0.0),
                'skip_probs': torch.ones(self.num_workers + 1),
                'aggregator_skip_prob': torch.tensor(1.0),
                'is_skipped': True,
                'forced_skip': True,  # Distinguish from natural skip
                'prompt_tokens': 0,
                'completion_tokens': 0,
            }

        task = input.get('task', '')

        # Build enhanced task with predecessor info
        enhanced_task = self._build_enhanced_task(task, spatial_info, temporal_info)
        enhanced_input = {**input, 'task': enhanced_task}

        # Handle aggregator_only mode (baseline): skip workers, call aggregator directly
        if self.aggregator_only:
            return await self._forward_aggregator_only(input, enhanced_input, hard, is_training)

        # Step 1: Select LLMs based on strategy
        if self.fixed_llm:
            # Use fixed LLM for all positions (ablation mode)
            llm_names = [self.fixed_llm_name] * (self.num_workers + 1)
            # Get device from selector to ensure consistency
            device = next(self.selector.parameters()).device
            log_probs = torch.zeros(self.num_workers + 1, device=device)
            entropy = torch.tensor(0.0, device=device)
            skip_probs = torch.zeros(self.num_workers + 1, device=device)
        elif inference_strategy == "topN_rotation" and not is_training:
            llm_names, log_probs, entropy, skip_probs = self.select_llms_topN_rotation(task, top_n=rotation_top_n)
        else:
            llm_names, log_probs, entropy, skip_probs = self.select_llms(task, hard=hard, is_training=is_training)

        worker_llms = llm_names[:-1]  # First N are workers
        aggregator_llm = llm_names[-1]  # Last is aggregator

        # Extract aggregator skip probability
        aggregator_skip_prob = skip_probs[-1]

        # Step 2: Check if aggregator selected skip (for skip learning)
        if aggregator_llm == 'skip':
            # This node is skipped - return empty result without executing LLM calls
            self._last_llm_selection = llm_names
            self._last_output = ""
            self._last_cost = 0.0
            self._last_prompt_tokens = 0
            self._last_completion_tokens = 0

            return {
                'output': '',
                'log_probs': log_probs,
                'cost': 0.0,
                'llm_selection': llm_names,
                'entropy': entropy,
                'skip_probs': skip_probs,
                'aggregator_skip_prob': aggregator_skip_prob,
                'is_skipped': True,
                'prompt_tokens': 0,
                'completion_tokens': 0,
            }

        # Step 3: Create MultiNode with selected LLMs
        multi_node = self.create_multi_node(worker_llms, aggregator_llm)

        # Step 4: Track tokens and execute MultiNode
        cost_before = Cost.instance().value
        prompt_before = PromptTokens.instance().value
        completion_before = CompletionTokens.instance().value

        output = await multi_node._async_execute(enhanced_input, {}, {})

        cost_after = Cost.instance().value
        cost = cost_after - cost_before

        # Compute token counts
        prompt_tokens = int(PromptTokens.instance().value - prompt_before)
        completion_tokens = int(CompletionTokens.instance().value - completion_before)

        # Update tracking attributes (including tokens)
        self._last_llm_selection = llm_names
        self._last_output = output if isinstance(output, str) else str(output)
        self._last_cost = cost
        self._last_prompt_tokens = prompt_tokens
        self._last_completion_tokens = completion_tokens

        return {
            'output': output,
            'log_probs': log_probs,
            'cost': cost,
            'llm_selection': llm_names,
            'entropy': entropy,
            'skip_probs': skip_probs,
            'aggregator_skip_prob': aggregator_skip_prob,
            'is_skipped': False,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
        }

    def get_selected_llm_names(self, task: str, hard: bool = True) -> List[str]:
        """
        Get selected LLM names without executing.

        Useful for visualization and debugging.

        Args:
            task: Task string
            hard: If True, use argmax selection

        Returns:
            List of selected LLM names
        """
        if self.fixed_llm:
            return [self.fixed_llm_name] * (self.num_workers + 1)
        llm_names, _, _, _ = self.select_llms(task, hard=hard)
        return llm_names

    def get_selection_probs(self, task: str, is_training: bool = True) -> torch.Tensor:
        """
        Get selection probabilities for all positions.

        Args:
            task: Task string
            is_training: If True, clamp logits; if False, no clamp

        Returns:
            Probability tensor (N+1, K+1)
        """
        device = next(self.selector.parameters()).device
        llm_emb = self.embedder.get_llm_embeddings(device=device)

        # Get task metadata
        metadata = self.task_cache.get_or_extract(task, use_llm=self.use_llm_for_task_meta)

        # Get merged task+role embedding
        task_role_emb = self.embedder.get_task_role_embedding(
            task,
            role=self.role,
            role_description=self.role_description,
            device=device
        )

        # Get metadata embedding
        meta_emb = self.embedder.get_meta_embedding(metadata, device=device)

        # Number of positions = workers + aggregator
        num_positions = self.num_workers + 1

        return self.selector.get_selection_probs(
            num_positions, llm_emb,
            task_emb=task_role_emb,  # Now contains task + role
            meta_emb=meta_emb,        # Unchanged
            aggregator_mask=True,
            is_training=is_training,
        )

    def set_temperature(self, temperature: float):
        """Set Gumbel-Softmax temperature.

        Only sets temperature if this node owns the selector.
        When using a shared external selector, temperature should be
        set at the graph level instead.
        """
        if self._owns_selector:
            self.selector.set_temperature(temperature)

    @property
    def temperature(self) -> float:
        """Get current temperature."""
        return self.selector.temperature
