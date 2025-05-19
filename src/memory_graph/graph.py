"""Graphs that extract memories on a schedule."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langmem import create_memory_store_manager
from typing_extensions import Annotated, TypedDict

from memory_graph import configuration


class State(TypedDict):
    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


class ProcessorState(State):
    """Extractor state."""

    function_name: str


logger = logging.getLogger("memory")


@functools.lru_cache(maxsize=100)
def get_store_manager(function_name: str):
    configurable = configuration.Configuration.from_context()
    memory_config = next(
        conf for conf in configurable.memory_types if conf.name == function_name
    )

    kwargs: dict[str, Any] = {
        "enable_inserts": memory_config.update_mode == "insert",
    }
    if memory_config.system_prompt:
        kwargs["instructions"] = memory_config.system_prompt

    return create_memory_store_manager(
        configurable.model,
        namespace=("memories", "{user_id}", function_name),
        **kwargs,
    )


@task()
async def process_memory_type(state: ProcessorState) -> None:
    """Extract the user's state from the conversation and update the memory."""
    configurable = configuration.Configuration.from_context()
    store_manager = get_store_manager(state["function_name"])
    await store_manager.ainvoke(
        {"messages": state["messages"], "max_steps": configurable.max_extraction_steps},
        config={
            "configurable": {
                "model": configurable.model,
                "user_id": configurable.user_id,
            }
        },
    )


@entrypoint(config_schema=configuration.Configuration)
async def graph(state: State) -> None:
    """Iterate over all memory types in the configuration.

    It will route each memory type from configuration to the corresponding memory update node.

    The memory update nodes will be executed in parallel.
    """
    if not state["messages"]:
        raise ValueError("No messages provided")
    configurable = configuration.Configuration.from_context()
    await asyncio.gather(
        *[
            process_memory_type(
                ProcessorState(messages=state["messages"], function_name=v.name),
            )
            for v in configurable.memory_types
        ]
    )


__all__ = ["graph"]
