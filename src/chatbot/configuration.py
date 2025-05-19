"""Define the configurable parameters for the chat bot."""

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langgraph.config import get_config

from chatbot.prompts import SYSTEM_PROMPT


@dataclass(kw_only=True)
class ChatConfigurable:
    """The configurable fields for the chatbot."""

    user_id: str = "default-user"
    mem_assistant_id: str = (
        "memory_graph"  # update to the UUID if you configure a custom assistant
    )
    model: str = "anthropic:claude-3-5-sonnet-20240620"
    delay_seconds: int = 3  # For debouncing memory creation
    system_prompt: str = SYSTEM_PROMPT
    memory_types: Optional[list[dict]] = None
    """The memory_types for the memory assistant."""

    @classmethod
    def from_context(cls) -> "ChatConfigurable":
        """Create a ChatConfigurable instance from a RunnableConfig object."""
        try:
            config = get_config()
            configurable = (
                config["configurable"] if config and "configurable" in config else {}
            )
        except RuntimeError:
            configurable = {}

        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in values.items() if v})
