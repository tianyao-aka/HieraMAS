from typing import Optional
from class_registry import ClassRegistry

from HieraMAS.llm.llm import LLM


class LLMRegistry:
    """
    Registry for LLM providers and models. The camera-ready code routes every
    model through OpenRouter; direct OpenAI and Together backends are not kept.

    OpenRouter Model Short Names:
        - qwen3-8b      -> qwen/qwen3-8b
        - qwen3-80b     -> qwen/qwen3-next-80b-a3b-instruct
        - deepseek-r1-14b -> deepseek/deepseek-r1-distill-qwen-32b
        - llama-8b      -> meta-llama/llama-3.1-8b-instruct
        - deepseek-v3   -> deepseek/deepseek-v3.2
    """
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        """
        Get an OpenRouter-backed LLM instance for the specified model.

        Args:
            model_name: The model name/ID. Can be:
                - None or empty: defaults to 'gpt-4o-mini'
                - OpenAI model name routed via OpenRouter, e.g. 'gpt-4o-mini'
                - OpenRouter short name: 'qwen3-8b', 'qwen3-80b', 'deepseek-r1-14b', 'llama-8b', 'deepseek-v3'
                - OpenRouter full ID: 'qwen/qwen3-8b', 'deepseek/deepseek-v3.2', etc.

        Returns:
            An LLM instance configured for the specified model

        Examples:
            >>> llm = LLMRegistry.get('gpt-4o')           # openai/gpt-4o via OpenRouter
            >>> llm = LLMRegistry.get('qwen3-8b')         # OpenRouter Qwen 3 8B
            >>> llm = LLMRegistry.get('llama-8b')         # OpenRouter Llama 8B
            >>> llm = LLMRegistry.get('deepseek-v3')      # OpenRouter DeepSeek V3
        """
        if model_name is None or model_name == "":
            model_name = "gpt-4o-mini"
        return cls.registry.get('OpenRouterChat', model_name)
    
    @classmethod
    def get_available_models(cls) -> dict:
        """
        Get a dictionary of all available models organized by provider.

        Returns:
            Dictionary with provider names as keys and lists of model names as values
        """
        return {
            "openrouter": [
                "gpt-4o-mini",     # openai/gpt-4o-mini via OpenRouter
                "gpt-4o",          # openai/gpt-4o via OpenRouter
                "gpt-5-mini",      # openai/gpt-5-mini via OpenRouter
                "gpt-5-nano",      # openai/gpt-5-nano via OpenRouter
                "qwen3-8b",         # qwen/qwen3-8b
                "qwen3-80b",        # qwen/qwen3-next-80b-a3b-instruct
                "deepseek-r1-14b",  # deepseek/deepseek-r1-distill-qwen-14b
                "llama-8b",         # meta-llama/llama-3.1-8b-instruct
                "deepseek-v3",      # deepseek/deepseek-v3.2
                "gemma-3-27b",      # google/gemma-3-27b-it
            ],
        }
