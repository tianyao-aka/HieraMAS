"""Compatibility chat helper backed only by OpenRouter.

Some experiment modules import `achat` for judge calls. Keep that import
surface stable, but route all requests through OpenRouter.
"""

from typing import Dict, List

from HieraMAS.llm.openrouter_chat import get_openrouter_model_id, openrouter_achat


async def achat(model: str, msg: List[Dict]):
    return await openrouter_achat(get_openrouter_model_id(model), msg)
