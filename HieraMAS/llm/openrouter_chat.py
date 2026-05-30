from openai import AsyncOpenAI
from typing import List, Union, Optional, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt, before_sleep_log
import logging

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
import os
import httpx

from HieraMAS.llm.format import Message
from HieraMAS.llm.llm import LLM
from HieraMAS.llm.llm_registry import LLMRegistry
from HieraMAS.utils.globals import Cost, PromptTokens, CompletionTokens, OpenRouterKeyVariant


load_dotenv()

# Global cache for model pricing (fetched from OpenRouter API)
_MODEL_PRICING_CACHE = {}
_PRICING_FETCHED = False

# Model short name mappings
OPENROUTER_MODELS = {
    # Qwen models
    "qwen3-8b": "qwen/qwen3-8b",
    "qwen3-80b": "qwen/qwen3-next-80b-a3b-instruct",
    # DeepSeek models
    "deepseek-r1-14b": "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek-v3": "deepseek/deepseek-v3.2",
    # Meta models
    "llama-8b": "meta-llama/llama-3.1-8b-instruct",
    # Google models
    "gemma-3-27b": "google/gemma-3-27b-it",
    # OpenAI models via OpenRouter
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
}

# Models that require minimum max_tokens >= 16
MODELS_MIN_TOKENS_16 = {"openai/gpt-5-mini", "openai/gpt-5-nano"}

OPENROUTER_MODELS_REVERSE = {v: k for k, v in OPENROUTER_MODELS.items()}

# Create async OpenAI client for OpenRouter
_client = None
_client_key_variant = None  # Track which variant the client was created with


def get_openrouter_api_key() -> str:
    """Get the appropriate OpenRouter API key based on configured variant.

    Returns:
        The API key string

    Raises:
        ValueError: If the required environment variable is not set
    """
    variant = OpenRouterKeyVariant.instance().value
    if variant == 2:
        key = os.getenv('OPENROUTER_API_KEY_2')
        if key is None:
            raise ValueError("OPENROUTER_API_KEY_2 environment variable not set but variant 2 requested")
        return key
    else:
        key = os.getenv('OPENROUTER_API_KEY')
        if key is None:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return key


def get_openrouter_base_url() -> str:
    """Get the appropriate OpenRouter base URL based on configured variant."""
    variant = OpenRouterKeyVariant.instance().value
    if variant == 2:
        return os.getenv('OPENROUTER_BASE_URL_2', 'https://openrouter.ai/api/v1')
    else:
        return os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')


def get_openrouter_client() -> AsyncOpenAI:
    """Get or create the AsyncOpenAI client for OpenRouter.

    The client is cached, but will be recreated if the key variant changes.
    """
    global _client, _client_key_variant

    current_variant = OpenRouterKeyVariant.instance().value

    if _client is None or _client_key_variant != current_variant:
        # Use custom httpx client with higher connection limits for concurrency
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            timeout=httpx.Timeout(120.0, connect=30.0),
        )
        _client = AsyncOpenAI(
            api_key=get_openrouter_api_key(),
            base_url=get_openrouter_base_url(),
            timeout=120.0,
            max_retries=0,  # Disable SDK retries; tenacity handles retries
            http_client=http_client,
        )
        _client_key_variant = current_variant

    return _client


def get_openrouter_model_id(model_name: str) -> str:
    """
    Convert short name to full model ID if needed.

    Args:
        model_name: Short name or full model ID

    Returns:
        Full model ID for OpenRouter API
    """
    if '/' in model_name:
        return model_name
    return OPENROUTER_MODELS.get(model_name.lower(), model_name)


def is_openrouter_model(model_name: str) -> bool:
    """
    Check if a model name corresponds to an OpenRouter model.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is an OpenRouter model, False otherwise
    """

    if model_name is None:
        return False

    # Check if it's a short name we recognize
    if model_name.lower() in OPENROUTER_MODELS:
        return True

    # Check if it's a full model ID we recognize
    if model_name in OPENROUTER_MODELS_REVERSE:
        return True

    # Check known prefixes for OpenRouter models
    openrouter_prefixes = ['qwen/', 'deepseek/', 'meta-llama/', 'google/', 'openai/']
    for prefix in openrouter_prefixes:
        if model_name.lower().startswith(prefix):
            return True

    return False


def fetch_openrouter_pricing_sync():
    """
    Fetch model pricing from OpenRouter API (sync, called once at startup).

    Raises:
        RuntimeError: If pricing cannot be fetched
    """
    global _MODEL_PRICING_CACHE, _PRICING_FETCHED

    if _PRICING_FETCHED:
        return

    try:
        # Use official OpenRouter API for pricing (proxy doesn't support /models endpoint)
        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {get_openrouter_api_key()}"},
            timeout=60.0,
        )
        response.raise_for_status()

        data = response.json()
        for model in data.get('data', []):
            model_id = model.get('id')
            pricing = model.get('pricing', {})
            if model_id and pricing:
                # API returns USD per token; multiply by 1000 for per-1K format
                _MODEL_PRICING_CACHE[model_id] = {
                    'prompt': float(pricing.get('prompt', 0)) * 1000,
                    'completion': float(pricing.get('completion', 0)) * 1000,
                }

        _PRICING_FETCHED = True
        print(f"[OpenRouter] Fetched pricing for {len(_MODEL_PRICING_CACHE)} models")

    except Exception as e:
        raise RuntimeError(f"[OpenRouter] Failed to fetch pricing: {e}")


def ensure_pricing_loaded():
    """Ensure pricing is loaded before cost calculation."""
    if not _PRICING_FETCHED:
        fetch_openrouter_pricing_sync()


async def openrouter_cost_count(model: str, prompt_tokens: int, completion_tokens: int):
    """
    Calculate and track cost using cached pricing from OpenRouter API (async, thread-safe).

    Args:
        model: The model ID
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Tuple of (cost, prompt_tokens, completion_tokens)

    Raises:
        RuntimeError: If pricing not found for model
    """
    ensure_pricing_loaded()

    pricing = _MODEL_PRICING_CACHE.get(model)
    if pricing is None:
        raise RuntimeError(f"[OpenRouter] No pricing found for model: {model}")

    # Pricing is USD per 1K tokens
    cost = (prompt_tokens * pricing['prompt'] / 1000 +
            completion_tokens * pricing['completion'] / 1000)

    # Update global counters (thread-safe async)
    await Cost.instance().add(cost)
    await PromptTokens.instance().add(prompt_tokens)
    await CompletionTokens.instance().add(completion_tokens)

    return cost, prompt_tokens, completion_tokens


@retry(
    wait=wait_random_exponential(max=60),  # Reduced from 240s
    stop=stop_after_attempt(4),  # Reduced from 6
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def openrouter_achat(
    model: str,
    msg: List[Dict],
    max_tokens: int = 1292,
    temperature: float = 0.2,
    num_comps: int = 1,
) -> Union[str, List[str]]:
    """
    Async chat completion with OpenRouter API using OpenAI SDK.

    Args:
        model: The model ID to use
        msg: List of message dictionaries
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_comps: Number of completions to generate

    Returns:
        The model's response content (str if num_comps=1, List[str] otherwise)
    """
    # Enforce minimum max_tokens for certain models
    if model in MODELS_MIN_TOKENS_16 and max_tokens < 16:
        max_tokens = 16

    client = get_openrouter_client()

    # Convert messages to proper format if needed
    messages = []
    for item in msg:
        if isinstance(item, dict):
            messages.append(item)
        else:
            # Handle Message dataclass objects
            messages.append({'role': item.role, 'content': item.content})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=num_comps,
    )

    # Extract usage from response and calculate cost
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        await openrouter_cost_count(model, prompt_tokens, completion_tokens)

    # Return result(s)
    if num_comps == 1:
        return response.choices[0].message.content
    else:
        return [choice.message.content for choice in response.choices]


@LLMRegistry.register('OpenRouterChat')
class OpenRouterChat(LLM):
    """
    OpenRouter chat implementation.

    Supports models via OpenRouter API:
    - qwen3-8b: qwen/qwen3-8b
    - qwen3-80b: qwen/qwen3-next-80b-a3b-instruct
    - deepseek-r1-14b: deepseek/deepseek-r1-distill-qwen-32b
    - llama-8b: meta-llama/llama-3.1-8b-instruct
    - deepseek-v3: deepseek/deepseek-v3.2
    """

    def __init__(self, model_name: str):
        self.model_name = get_openrouter_model_id(model_name)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await openrouter_achat(
            self.model_name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_comps=num_comps,
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
