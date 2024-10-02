"""Define the shared values."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass(kw_only=True)
class State:
    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


@dataclass(kw_only=True)
class ProcessorState(State):
    """Extractor state."""

    function_name: str


__all__ = [
    "State",
    "ProcessorState",
]
