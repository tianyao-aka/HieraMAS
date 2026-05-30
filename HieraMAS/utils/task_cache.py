"""
Task Metadata Cache for Learnable LLM Selection.

This module caches task metadata to avoid redundant LLM calls during training.
Task metadata includes: type, difficulty, domain, description, and key_skills.
"""

import hashlib
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

from HieraMAS.utils.const import HIERAMAS_ROOT


class TaskMetaCache:
    """
    Caches task metadata extracted via LLM to avoid redundant API calls.

    The cache is stored as a JSON file, keyed by task hash.
    """

    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the task metadata cache.

        Args:
            cache_file: Path to cache file. Defaults to {HIERAMAS_ROOT}/math/cache/task_meta_cache.json
        """
        if cache_file is None:
            cache_dir = Path(HIERAMAS_ROOT) / "math" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / "task_meta_cache.json"
        else:
            self.cache_file = Path(cache_file)

        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save task cache: {e}")

    def _hash_task(self, task: str) -> str:
        """Generate a hash key for a task string."""
        return hashlib.md5(task.encode('utf-8')).hexdigest()

    def get_or_extract(self, task: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Get cached metadata or extract via LLM.

        Args:
            task: The task string
            use_llm: Whether to use LLM for extraction (if False, returns heuristic)

        Returns:
            Task metadata dict with keys: type, difficulty, domain, description, key_skills
        """
        task_hash = self._hash_task(task)

        if task_hash in self.cache:
            return self.cache[task_hash]

        # Extract metadata
        if use_llm:
            metadata = self._extract_metadata_llm(task)
        else:
            metadata = self._extract_metadata_heuristic(task)

        # Cache and save
        self.cache[task_hash] = metadata
        self._save_cache()

        return metadata

    def _extract_metadata_heuristic(self, task: str) -> Dict[str, Any]:
        """
        Extract task metadata using simple heuristics (no LLM call).

        This is a fallback for when LLM calls should be avoided.
        """
        task_lower = task.lower()

        # Detect task type
        if any(kw in task_lower for kw in ['calculate', 'compute', 'solve', 'equation', 'math']):
            task_type = 'math'
            key_skills = ['mathematical reasoning', 'calculation', 'step-by-step solving']
        elif any(kw in task_lower for kw in ['code', 'function', 'program', 'implement', 'python', 'def ']):
            task_type = 'coding'
            key_skills = ['programming', 'code generation', 'debugging']
        elif any(kw in task_lower for kw in ['explain', 'describe', 'what is', 'define']):
            task_type = 'explanation'
            key_skills = ['knowledge retrieval', 'clear explanation']
        elif any(kw in task_lower for kw in ['analyze', 'compare', 'evaluate', 'reason']):
            task_type = 'analysis'
            key_skills = ['critical thinking', 'logical reasoning', 'analysis']
        else:
            task_type = 'general'
            key_skills = ['reasoning', 'problem-solving']

        # Estimate difficulty by length and complexity indicators
        word_count = len(task.split())
        if word_count > 200 or any(kw in task_lower for kw in ['complex', 'advanced', 'difficult']):
            difficulty = 'hard'
        elif word_count > 50:
            difficulty = 'medium'
        else:
            difficulty = 'easy'

        return {
            'type': task_type,
            'difficulty': difficulty,
            'domain': task_type,  # Use type as domain for heuristic
            'description': task[:200] if len(task) > 200 else task,
            'key_skills': key_skills,
        }

    def _extract_metadata_llm(self, task: str) -> Dict[str, Any]:
        """
        Extract task metadata using LLM (gpt-4o-mini).

        Falls back to heuristic if LLM call fails.
        """
        try:
            from HieraMAS.llm.llm_registry import LLMRegistry
            import asyncio

            llm = LLMRegistry.get('gpt-4o-mini')

            prompt = f"""Analyze this task and extract metadata in JSON format.

Task:
{task[:1500]}

Return ONLY a JSON object with these fields:
- type: one of [math, coding, analysis, explanation, reasoning, general]
- difficulty: one of [easy, medium, hard]
- domain: specific domain (e.g., algebra, python, physics)
- description: one-sentence summary
- key_skills: list of 2-4 required skills

Example output:
{{"type": "math", "difficulty": "medium", "domain": "algebra", "description": "Solve a quadratic equation", "key_skills": ["algebraic manipulation", "equation solving"]}}

JSON output:"""

            messages = [
                {'role': 'system', 'content': 'You are a task analyzer. Output only valid JSON.'},
                {'role': 'user', 'content': prompt},
            ]

            # Run async LLM call
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, llm.agen(messages, max_tokens=256))
                    response = future.result(timeout=30)
            else:
                response = asyncio.run(llm.agen(messages, max_tokens=256))

            # Parse JSON from response
            response_text = response.strip()
            # Handle potential markdown code blocks
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            metadata = json.loads(response_text)

            # Validate required fields
            required = ['type', 'difficulty', 'domain', 'description', 'key_skills']
            if all(k in metadata for k in required):
                return metadata

        except Exception as e:
            print(f"LLM metadata extraction failed: {e}, using heuristic")

        # Fallback to heuristic
        return self._extract_metadata_heuristic(task)

    def clear_cache(self):
        """Clear all cached metadata."""
        self.cache = {}
        self._save_cache()

    def __len__(self) -> int:
        """Return number of cached tasks."""
        return len(self.cache)
