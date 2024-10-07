"""Graphs that extract memories on a schedule."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send
from trustcall import create_extractor

from memory_graph import configuration, utils
from memory_graph.state import ProcessorState, State

logger = logging.getLogger("memory")


async def handle_patch_memory(
    state: ProcessorState, config: RunnableConfig, *, store: BaseStore
) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    # Get the overall configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # Namespace for memory events, where function_name is the name of the memory schema
    namespace = (configurable.user_id, "user_states")

    # Fetch existing memories from the store for this (patch) memory schema
    existing_item = await store.aget(namespace, state.function_name)
    existing = {state.function_name: existing_item.value} if existing_item else None

    # Get the configuration for this memory schema (identified by function_name)
    memory_config = next(
        conf for conf in configurable.memory_types if conf.name == state.function_name
    )

    # This is what we use to generate new memories
    extractor = create_extractor(
        utils.init_model(configurable.model),
        # We pass the specified (patch) memory schema as a tool
        tools=[
            {
                # Tool name
                "name": memory_config.name,
                # Tool description
                "description": memory_config.description,
                # Schema for patch memory
                "parameters": memory_config.parameters,
            }
        ],
        tool_choice=memory_config.name,
    )

    # Prepare the messages
    prepared_messages = utils.prepare_messages(
        state.messages, memory_config.system_prompt
    )

    # Pass messages and existing patch to the extractor
    inputs = {"messages": prepared_messages, "existing": existing}
    # Update the patch memory
    result = await extractor.ainvoke(inputs, config)
    extracted = result["responses"][0].model_dump(mode="json")
    # Save to storage
    await store.aput(namespace, state.function_name, extracted)


async def handle_insertion_memory(
    state: ProcessorState, config: RunnableConfig, *, store: BaseStore
) -> dict[str, list]:
    """Handle insertion memory events."""
    # Get the overall configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # Namespace for memory events, where function_name is the name of the memory schema
    namespace = (configurable.user_id, "events", state.function_name)

    # Fetch existing memories from the store (5 most recent ones) for the this (insert) memory schema
    existing_items = await store.asearch(namespace, limit=5)

    # Get the configuration for this memory schema (identified by function_name)
    memory_config = next(
        conf for conf in configurable.memory_types if conf.name == state.function_name
    )

    # This is what we use to generate new memories
    extractor = create_extractor(
        utils.init_model(configurable.model),
        # We pass the specified (insert) memory schema as a tool
        tools=[
            {
                # Tool name
                "name": memory_config.name,
                # Tool description
                "description": memory_config.description,
                # Schema for insert memory
                "parameters": memory_config.parameters,
            }
        ],
        tool_choice="any",
        # This allows the extractor to insert new memories
        enable_inserts=True,
    )

    # Generate new memories or update existing memories
    extracted = await extractor.ainvoke(
        {
            # Prepare the messages
            "messages": utils.prepare_messages(
                state.messages, memory_config.system_prompt
            ),
            # Prepare the existing memories
            "existing": (
                [
                    (existing_item.key, state.function_name, existing_item.value)
                    for existing_item in existing_items
                ]
                if existing_items
                else None
            ),
        },
        config,
    )

    # Add the memories to storage
    await asyncio.gather(
        *(
            store.aput(
                namespace,
                rmeta.get("json_doc_id", str(uuid.uuid4())),
                r.model_dump(mode="json"),
            )
            for r, rmeta in zip(extracted["responses"], extracted["response_metadata"])
        )
    )


# Create the graph and all nodes
builder = StateGraph(State, config_schema=configuration.Configuration)
builder.add_node(handle_patch_memory, input=ProcessorState)
builder.add_node(handle_insertion_memory, input=ProcessorState)


def scatter_schemas(state: State, config: RunnableConfig) -> list[Send]:
    """Iterate over all memory types in the configuration.

    It will route each memory type from configuration to the corresponding memory update node.

    The memory update nodes will be executed in parallel.
    """
    # Get the configuration
    configurable = configuration.Configuration.from_runnable_config(config)
    sends = []
    current_state = asdict(state)

    # Loop over all memory types specified in the configuration
    for v in configurable.memory_types:
        update_mode = v.update_mode

        # This specifies the type of memory update to perform from the configuration
        match update_mode:
            case "patch":
                # This is the corresponding node in the graph for the patch-based memory update
                target = "handle_patch_memory"
            case "insert":
                # This is the corresponding node in the graph for the insert-based memory update
                target = "handle_insertion_memory"
            case _:
                raise ValueError(f"Unknown update mode: {update_mode}")

        # Use Send API to route to the target node and pass the name of the memory schema as function_name
        # Send API allows each memory node to be executed in parallel
        sends.append(
            Send(
                target,
                ProcessorState(**{**current_state, "function_name": v.name}),
            )
        )
    return sends


# Add conditional edges to the graph
builder.add_conditional_edges(
    "__start__", scatter_schemas, ["handle_patch_memory", "handle_insertion_memory"]
)

# Compile the graph
graph = builder.compile()
__all__ = ["graph"]
