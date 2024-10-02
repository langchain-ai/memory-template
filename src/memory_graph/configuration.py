"""Define the configurable parameters for the memory service."""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated


@dataclass(kw_only=True)
class MemoryConfig:
    """Configuration for memory-related operations."""

    name: str
    """This tells the model how to reference the function 
    
    and organizes related memories within the namespace."""
    description: str
    """Description for what this memory type is intended to capture."""

    parameters: dict[str, Any]
    """The JSON Schema of the memory document to manage."""
    system_prompt: str = ""
    """The system prompt to use for the memory assistant."""
    update_mode: Literal["patch", "insert"] = field(default="patch")
    """Whether to continuously patch the memory, or treat each new

    generation as a new memory.

    Patching is useful for maintaining a structured profile or core list
    of memories. Inserting is useful for maintaining all interactions and
    not losing any information.

    For patched memories, you can GET the current state at any given time.
    For inserted memories, you can query the full history of interactions.
    """


@dataclass(kw_only=True)
class Configuration:
    """Main configuration class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

    """The model to use for generating memories. """
    memory_types: list[MemoryConfig] = field(default_factory=list)
    """The memory_types for the memory assistant."""

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        if values.get("memory_types") is None:
            values["memory_types"] = DEFAULT_MEMORY_CONFIGS.copy()
        else:
            values["memory_types"] = [
                MemoryConfig(**v) for v in (values["memory_types"] or [])
            ]
        return cls(**{k: v for k, v in values.items() if v})


DEFAULT_MEMORY_CONFIGS = [
    MemoryConfig(
        name="User",
        description="Update this document to maintain up-to-date information about the user in the conversation.",
        update_mode="patch",
        parameters={
            "type": "object",
            "properties": {
                "user_name": {
                    "type": "string",
                    "description": "The user's preferred name",
                },
                "age": {"type": "integer", "description": "The user's age"},
                "interests": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of the user's interests",
                },
                "home": {
                    "type": "string",
                    "description": "Description of the user's home town/neighborhood, etc.",
                },
                "occupation": {
                    "type": "string",
                    "description": "The user's current occupation or profession",
                },
                "conversation_preferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of the user's preferred conversation styles, pronouns, topics they want to avoid, etc.",
                },
            },
        },
    ),
    MemoryConfig(
        name="Note",
        description="Save notable memories the user has shared with you for later recall.",
        update_mode="insert",
        parameters={
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The situation or circumstance where this memory may be relevant. "
                    "Include any caveats or conditions that contextualize the memory. "
                    "For example, if a user shares a preference, note if it only applies "
                    "in certain situations (e.g., 'only at work'). Add any other relevant "
                    "'meta' details that help fully understand when and how to use this memory.",
                },
                "content": {
                    "type": "string",
                    "description": "The specific information, preference, or event being remembered.",
                },
            },
            "required": ["context", "content"],
        },
    ),
]
