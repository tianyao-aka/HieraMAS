import sys
import random
import asyncio
from typing import Union, Literal, List

class Singleton:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self):
        self.value = 0.0


class AsyncLockSingleton(Singleton):
    """Singleton with asyncio.Lock for thread-safe async updates."""

    def __init__(self):
        self.value = 0.0
        self._lock = asyncio.Lock()

    async def add(self, amount: float):
        """Thread-safe async addition to value."""
        async with self._lock:
            self.value += amount

    def add_sync(self, amount: float):
        """Synchronous addition (use only when not in async context)."""
        self.value += amount


class Cost(AsyncLockSingleton):
    def __init__(self):
        super().__init__()


class PromptTokens(AsyncLockSingleton):
    def __init__(self):
        super().__init__()


class CompletionTokens(AsyncLockSingleton):
    def __init__(self):
        super().__init__()
        self.value = 0  # Integer for token counts

class Time(Singleton):
    def __init__(self):
        self.value = ""

class Mode(Singleton):
    def __init__(self):
        self.value = ""


class OpenRouterKeyVariant(Singleton):
    """Singleton to manage which OpenRouter API key variant to use.

    Variant 1 (default): Uses OPENROUTER_API_KEY environment variable
    Variant 2: Uses OPENROUTER_API_KEY_2 environment variable
    """
    def __init__(self):
        self.value = 1

    def reset(self):
        self.value = 1
