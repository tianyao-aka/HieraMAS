"""
LLM Profiles for Learnable LLM Selection.

This module contains detailed text descriptions of each LLM option
for embedding-based selection. The profiles describe model capabilities,
strengths, weaknesses, and cost characteristics.
"""

from typing import Dict, List


# Ordered list of LLM options (index corresponds to selection)
LLM_OPTIONS: List[str] = [
    'qwen3-8b',
    'qwen3-80b',
    'deepseek-r1-14b',
    'llama-8b',
    'deepseek-v3',
    'gemma-3-27b',
    'gpt-5-nano',
    'gpt-5-mini',
    'gpt-4o-mini',
    'skip',  # Special token: no LLM call for this position
]


# Detailed text profiles for each LLM (used for embedding)
LLM_PROFILES: Dict[str, str] = {
    'qwen3-8b': """
Qwen 3 8B is a fast and efficient 8 billion parameter language model from Alibaba.
Strengths: Very fast inference, low latency, cost-effective for simple tasks.
Good for: Basic text generation, simple Q&A, straightforward reasoning tasks.
Weaknesses: Limited reasoning depth, may struggle with complex multi-step problems.
Cost tier: Very low cost, suitable for high-volume simple tasks.
Best used when: Speed and cost matter more than reasoning depth.
""".strip(),

    'qwen3-80b': """
Qwen 3 80B is a powerful 80 billion parameter mixture-of-experts model from Alibaba.
Strengths: Strong reasoning, excellent at complex tasks, good at math and coding.
Good for: Complex reasoning, mathematical problem-solving, code generation.
Weaknesses: Higher latency and cost compared to smaller models.
Cost tier: Medium-high cost, good value for complex tasks.
Best used when: Task requires deep reasoning or complex problem-solving.
""".strip(),

    'deepseek-r1-14b': """
DeepSeek R1 14B is a math-focused 14 billion parameter model optimized for reasoning.
Strengths: Excellent mathematical reasoning, step-by-step problem solving.
Good for: Math problems, logical reasoning, structured analytical tasks.
Weaknesses: May be less versatile for general text generation tasks.
Cost tier: Low-medium cost, excellent value for math-heavy workloads.
Best used when: Task involves mathematical computation or formal reasoning.
""".strip(),

    'llama-8b': """
Llama 3.1 8B is a general-purpose 8 billion parameter model from Meta.
Strengths: Well-balanced capabilities, good instruction following, reliable.
Good for: General tasks, conversation, basic reasoning, text processing.
Weaknesses: Not specialized for any particular domain.
Cost tier: Very low cost, good baseline model.
Best used when: Need a reliable, fast, general-purpose model.
""".strip(),

    'deepseek-v3': """
DeepSeek V3.2 is a state-of-the-art reasoning model with excellent performance.
Strengths: Top-tier reasoning, excellent at complex problems, strong coding ability.
Good for: Complex reasoning, advanced math, sophisticated code generation.
Weaknesses: Higher latency, may be overkill for simple tasks.
Cost tier: Medium cost, excellent performance-to-cost ratio.
Best used when: Task requires state-of-the-art reasoning capabilities.
""".strip(),

    'gemma-3-27b': """
Gemma 3 27B is a balanced 27 billion parameter model from Google.
Strengths: Good balance of speed and capability, strong instruction following.
Good for: Medium complexity tasks, balanced workloads, general reasoning.
Weaknesses: Not the best at either extreme (speed or capability).
Cost tier: Medium cost, good middle-ground option.
Best used when: Need balance between speed, cost, and capability.
""".strip(),

    'gpt-5-nano': """
GPT-5 Nano is an extremely fast and cheap model from OpenAI.
Strengths: Very low latency, lowest cost, good for simple tasks.
Good for: Simple text processing, basic Q&A, high-volume simple tasks.
Weaknesses: Limited reasoning capability, may fail on complex problems.
Cost tier: Lowest cost option available.
Best used when: Speed and cost are critical, task is simple.
""".strip(),

    'gpt-5-mini': """
GPT-5 Mini is a capable model from OpenAI with strong reasoning abilities.
Strengths: Strong reasoning, excellent at complex tasks, reliable outputs.
Good for: Complex reasoning, multi-step problems, advanced coding tasks.
Weaknesses: Higher cost than smaller models, may be overkill for simple tasks.
Cost tier: Medium-high cost ($0.25/1M input, $2.00/1M output).
Best used when: Task requires strong reasoning and quality matters more than cost.
""".strip(),

    'gpt-4o-mini': """
GPT-4o Mini is a capable and reliable model from OpenAI.
Strengths: Strong reasoning, good at coding, reliable instruction following.
Good for: Complex reasoning, code generation, analytical tasks.
Weaknesses: Higher cost than smaller models.
Cost tier: Medium-high cost, proven reliability.
Best used when: Need proven, reliable performance on complex tasks.
""".strip(),

    'skip': """
Skip token: Do not assign any LLM to this position.
This option removes this agent from the pipeline entirely.
Effect: No API call made, no cost incurred, no output from this position.
Use when: This position is not needed for the current task.
Useful for: Reducing cost when fewer agents are sufficient.
Warning: Skipping too many positions may reduce solution quality.
""".strip(),
}


# Per-1K-token costs for reward calculation (approximate)
LLM_COSTS: Dict[str, float] = {
    'qwen3-8b': 0.0001,
    'qwen3-80b': 0.001,
    'deepseek-r1-14b': 0.0003,
    'llama-8b': 0.0001,
    'deepseek-v3': 0.0005,
    'gemma-3-27b': 0.0004,
    'gpt-5-nano': 0.00005,
    'gpt-5-mini': 0.001,   # avg of $0.25/1M input + $2.00/1M output
    'gpt-4o-mini': 0.0006,
    'skip': 0.0,
}


def get_llm_index(llm_name: str) -> int:
    """Get the index of an LLM in LLM_OPTIONS."""
    return LLM_OPTIONS.index(llm_name)


def get_llm_name(index: int) -> str:
    """Get the LLM name at a given index."""
    return LLM_OPTIONS[index]


def get_num_llms() -> int:
    """Get the total number of LLM options (including skip)."""
    return len(LLM_OPTIONS)
