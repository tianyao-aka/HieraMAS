"""
MultiNode: A composite node containing multiple instances of the same agent type.

This module provides the MultiNode class which enables multiple agents with the
same role to work independently on a problem, with an aggregator node that
synthesizes their outputs into a final answer.
"""

import asyncio
from typing import List, Any, Optional, Dict

from HieraMAS.graph.node import Node
from HieraMAS.agents.agent_registry import AgentRegistry


class MultiNode(Node):
    """
    A composite node with multiple agent instances of the same type + aggregator.

    Workers process independently (no inter-worker communication), and the
    aggregator synthesizes their outputs into a final answer.

    Architecture:
        Input -> [Worker 1] -\
              -> [Worker 2] ---> [Aggregator] -> Output
              -> [Worker N] -/

    Attributes:
        agent_name (str): The name of the agent type for workers (e.g., "MathSolver")
        num_agents (int): Number of worker instances
        role (str): The role all workers share
        worker_nodes (List[Node]): List of worker agent instances
        aggregator_node (Node): The aggregator node that combines outputs

    Example:
        >>> multi_node = MultiNode(
        ...     id="math_team",
        ...     agent_name="MathSolver",
        ...     num_agents=5,
        ...     domain="gsm8k",
        ...     llm_name="gpt-4o",
        ...     role="Math Solver"
        ... )
        >>> result = multi_node.execute({'task': 'What is 2 + 2?'})
    """

    def __init__(
        self,
        id: Optional[str] = None,
        agent_name: str = "MathSolver",
        num_agents: int = 3,
        aggregator_method: str = "FinalMultiAgentAggregator",
        domain: str = "gsm8k",
        llm_name: str = "gpt-4o-mini",
        role: str = "Math Solver",
        worker_llm_names: Optional[List[str]] = None,
        aggregator_llm_name: Optional[str] = None,
    ):
        """
        Initialize a MultiNode with multiple worker agents and an aggregator.

        Args:
            id: Unique identifier for this MultiNode (auto-generated if None)
            agent_name: Name of the registered agent type for workers
            num_agents: Number of worker instances to create
            aggregator_method: Name of the registered aggregator agent type
            domain: Domain context (e.g., "gsm8k", "humaneval")
            llm_name: Default LLM name (fallback if worker_llm_names not specified)
            role: The role all workers share (e.g., "Math Solver")
            worker_llm_names: List of LLM names for each worker (length must match num_agents)
            aggregator_llm_name: LLM name for the aggregator (defaults to llm_name)
        """
        super().__init__(id, "MultiNode", domain, llm_name)

        self.agent_name = agent_name
        self.num_agents = num_agents
        self.role = role  # All workers share this role
        self.worker_nodes: List[Node] = []
        self.aggregator_node: Node = None

        # Handle LLM assignments
        default_llm = llm_name or "gpt-4o-mini"

        # Worker LLMs: use provided list, or default for all
        if worker_llm_names is not None:
            if len(worker_llm_names) != num_agents:
                raise ValueError(
                    f"worker_llm_names length ({len(worker_llm_names)}) "
                    f"must match num_agents ({num_agents})"
                )
            self.worker_llm_names = worker_llm_names
        else:
            self.worker_llm_names = [default_llm] * num_agents

        # Aggregator LLM: use provided, or default
        self.aggregator_llm_name = aggregator_llm_name or default_llm

        # Initialize internal structure
        self._init_workers(role)
        self._init_aggregator(aggregator_method)
        self._connect_internal()

    def _init_workers(self, role: str):
        """
        Create N independent instances of the same agent type with the same role.

        Each worker can have a different LLM backbone based on worker_llm_names.

        Args:
            role: The role for all worker instances
        """
        for i in range(self.num_agents):
            worker = AgentRegistry.get(
                self.agent_name,
                id=f"{self.id}_w{i}",
                role=role,
                domain=self.domain,
                llm_name=self.worker_llm_names[i],  # Per-worker LLM
            )
            self.worker_nodes.append(worker)

    def _init_aggregator(self, method: str):
        """
        Create the aggregator node that will combine worker outputs.

        The aggregator can have a different LLM backbone from workers.

        Args:
            method: Name of the registered aggregator agent type
        """
        self.aggregator_node = AgentRegistry.get(
            method,
            id=f"{self.id}_agg",
            domain=self.domain,
            llm_name=self.aggregator_llm_name,  # Aggregator-specific LLM
            worker_role=self.role,  # Pass worker role for role-specific aggregation
        )

    def _connect_internal(self):
        """
        Connect all workers as spatial predecessors to the aggregator.

        This creates the internal graph structure:
        Worker 1 -> Aggregator
        Worker 2 -> Aggregator
        ...
        Worker N -> Aggregator
        """
        for worker in self.worker_nodes:
            self.aggregator_node.add_predecessor(worker, st='spatial')

    def _process_inputs(
        self,
        raw_inputs: Dict[str, Any],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process inputs for the multi-node.

        For MultiNode, we simply pass through the raw inputs as workers
        will process them independently.
        """
        return raw_inputs

    def _execute(
        self,
        input: Dict[str, Any],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Execute the multi-node synchronously.

        Steps:
        1. Execute all workers independently (no inter-worker communication)
        2. Aggregator collects all worker outputs via spatial connections
        3. Return aggregator's final synthesized output

        Args:
            input: The input task (typically {'task': '...'})
            spatial_info: Spatial info from external predecessors (not used internally)
            temporal_info: Temporal info from previous rounds (not used internally)

        Returns:
            The aggregator's final output
        """
        # Step 1: Execute all workers independently
        for worker in self.worker_nodes:
            worker.execute(input, **kwargs)

        # Step 2: Aggregator gets all worker outputs via get_spatial_info()
        aggregator_output = self.aggregator_node.execute(input, **kwargs)

        return aggregator_output[0] if aggregator_output else ""

    async def _async_execute(
        self,
        input: Dict[str, Any],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Execute the multi-node asynchronously with parallel worker execution.

        Workers are executed concurrently using asyncio.gather for better performance.

        Args:
            input: The input task (typically {'task': '...'})
            spatial_info: Spatial info from external predecessors (not used internally)
            temporal_info: Temporal info from previous rounds (not used internally)

        Returns:
            The aggregator's final output
        """
        # Step 1: Execute all workers in parallel
        worker_tasks = [
            worker.async_execute(input, **kwargs)
            for worker in self.worker_nodes
        ]
        await asyncio.gather(*worker_tasks)

        # Step 2: Execute aggregator
        aggregator_output = await self.aggregator_node.async_execute(input, **kwargs)

        return aggregator_output[0] if aggregator_output else ""

    def get_internal_nodes(self) -> List[Node]:
        """
        Get all internal nodes (workers + aggregator).

        Useful for debugging and visualization.

        Returns:
            List of all internal Node instances
        """
        return self.worker_nodes + [self.aggregator_node]

    def get_worker_outputs(self) -> List[Any]:
        """
        Get outputs from all worker nodes.

        Useful for inspecting individual worker contributions.

        Returns:
            List of outputs from each worker
        """
        return [worker.outputs for worker in self.worker_nodes]
