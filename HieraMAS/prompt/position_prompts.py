"""
Position Prompt Generator for Learnable LLM Selection.

This module generates task-specific position descriptions for embedding.
Each position (worker or aggregator) gets a description that captures
its role in the multi-agent pipeline and the task context.
"""

from typing import List, Dict, Any, Optional


class PositionPromptGenerator:
    """Generates position-specific prompts for embedding-based LLM selection."""

    @staticmethod
    def worker_prompt(
        idx: int,
        total: int,
        task_info: Dict[str, Any],
        role: Optional[str] = None,
        role_description: Optional[str] = None,
    ) -> str:
        """
        Generate a description for a worker position.

        Args:
            idx: Worker index (0-based)
            total: Total number of workers
            task_info: Task metadata (type, difficulty, domain, etc.)
            role: Optional role name (e.g., "Math Solver")
            role_description: Optional detailed description of the role

        Returns:
            A text description of this worker position
        """
        task_type = task_info.get('type', 'general')
        difficulty = task_info.get('difficulty', 'medium')
        domain = task_info.get('domain', 'general')
        description = task_info.get('description', '')
        key_skills = task_info.get('key_skills', [])

        skills_str = ', '.join(key_skills) if key_skills else 'general reasoning'

        # Build role section if role is provided
        role_section = ""
        if role:
            role_section = f"""
Role: {role}
Role responsibilities: {role_description or 'General problem solving'}
"""

        prompt = f"""
Worker agent position {idx + 1} of {total} in a multi-agent problem-solving pipeline.
{role_section}

Task context:
- Task type: {task_type}
- Difficulty level: {difficulty}
- Domain: {domain}
- Key skills required: {skills_str}

This worker operates independently and provides its solution to the aggregator.
Workers do not communicate with each other directly.
The worker should focus on generating a complete, well-reasoned solution.

Task description excerpt: {description[:200] if description else 'Not available'}
""".strip()

        return prompt

    @staticmethod
    def aggregator_prompt(
        num_workers: int,
        task_info: Dict[str, Any],
        role: Optional[str] = None,
        role_description: Optional[str] = None,
    ) -> str:
        """
        Generate a description for the aggregator position.

        Args:
            num_workers: Number of worker agents
            task_info: Task metadata (type, difficulty, domain, etc.)
            role: Optional role name (e.g., "Math Solver")
            role_description: Optional detailed description of the role

        Returns:
            A text description of the aggregator position
        """
        task_type = task_info.get('type', 'general')
        difficulty = task_info.get('difficulty', 'medium')
        domain = task_info.get('domain', 'general')

        # Build role section if role is provided
        role_section = ""
        if role:
            role_section = f"""
Role: {role} (Aggregator)
Role responsibilities: {role_description or 'Synthesize worker outputs'}
"""

        prompt = f"""
Aggregator agent position in a multi-agent problem-solving pipeline.
{role_section}
Role: Synthesize and combine outputs from all active worker agents.

Task context:
- Task type: {task_type}
- Difficulty level: {difficulty}
- Domain: {domain}

The aggregator receives all worker solutions and must:
1. Analyze each worker's approach and reasoning
2. Identify correct insights and filter out errors
3. Synthesize the best elements into a final answer
4. Provide a coherent, well-reasoned final solution

This position requires strong analytical and synthesis capabilities.
The aggregator has access to all worker outputs and the original task.
""".strip()

        return prompt

    @staticmethod
    def get_all_position_prompts(
        num_workers: int,
        task_info: Dict[str, Any],
        role: Optional[str] = None,
        role_description: Optional[str] = None,
    ) -> List[str]:
        """
        Get prompts for all positions (workers + aggregator).

        Args:
            num_workers: Number of worker agents
            task_info: Task metadata
            role: Optional role name (e.g., "Math Solver")
            role_description: Optional detailed description of the role

        Returns:
            List of prompts: [worker_0, worker_1, ..., worker_n-1, aggregator]
        """
        prompts = []

        # Worker prompts
        for i in range(num_workers):
            prompts.append(
                PositionPromptGenerator.worker_prompt(
                    i, num_workers, task_info, role, role_description
                )
            )

        # Aggregator prompt
        prompts.append(
            PositionPromptGenerator.aggregator_prompt(
                num_workers, task_info, role, role_description
            )
        )

        return prompts


def create_default_task_info() -> Dict[str, Any]:
    """Create a default task info dict when metadata is not available."""
    return {
        'type': 'general',
        'difficulty': 'medium',
        'domain': 'general',
        'description': '',
        'key_skills': ['reasoning', 'problem-solving'],
    }
