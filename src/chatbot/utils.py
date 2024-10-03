"""Define utility functions for your graph."""

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.store.base import Item


def format_memories(memories: Optional[list[Item]]) -> str:
    """Format the user's memories."""
    if not memories:
        return ""
    # Note Bene: You can format better than this....
    formatted_memories = "\n".join(
        f"{str(m.value)}\tLast updated: {m.updated_at}" for m in memories
    )
    return f"""

## Memories

You have noted the following memorable events from previous interactions with the user.
<memories>
{formatted_memories}
</memories>
"""


def init_model(fully_specified_name: str) -> BaseChatModel:
    """Initialize the configured chat model."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)
