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
    configurable = configuration.Configuration.from_runnable_config(config)
    namespace = (configurable.user_id, "user_states", state.function_name)
    existing_item = await store.aget(namespace, "memory")
    existing = {existing_item.key: existing_item.value} if existing_item else None
    memory_config = next(
        conf for conf in configurable.memory_types if conf.name == state.function_name
    )
    extractor = create_extractor(
        utils.init_model(configurable.model),
        tools=[
            {
                "name": memory_config.name,
                "description": memory_config.description,
                "parameters": memory_config.parameters,
            }
        ],
        tool_choice=memory_config.name,
    )
    prepared_messages = utils.prepare_messages(
        state.messages, memory_config.system_prompt
    )
    inputs = {"messages": prepared_messages, "existing": existing}
    result = await extractor.ainvoke(inputs, config)
    extracted = result["responses"][0].model_dump(mode="json")
    # Upsert the memory to storage
    await store.aput(namespace, "memory", extracted)
    return {"messages": []}


async def handle_insertion_memory(
    state: ProcessorState, config: RunnableConfig, *, store: BaseStore
) -> dict[str, list]:
    """Upsert memory events."""
    configurable = configuration.Configuration.from_runnable_config(config)
    namespace = (configurable.user_id, "events", state.function_name)
    existing_items = await store.asearch(namespace, limit=5)
    memory_config = next(
        conf for conf in configurable.memory_types if conf.name == state.function_name
    )
    extractor = create_extractor(
        utils.init_model(configurable.model),
        tools=[
            {
                "name": memory_config.name,
                "description": memory_config.description,
                "parameters": memory_config.parameters,
            }
        ],
        tool_choice="any",
        enable_inserts=True,
    )
    extracted = await extractor.ainvoke(
        {
            "messages": utils.prepare_messages(
                state.messages, memory_config.system_prompt
            ),
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
    return {"messages": []}


# Create the graph + all nodes
builder = StateGraph(State, config_schema=configuration.Configuration)

builder.add_node(handle_patch_memory, input=ProcessorState)
builder.add_node(handle_insertion_memory, input=ProcessorState)


def scatter_schemas(state: State, config: RunnableConfig) -> list[Send]:
    """Route the memory_types for the memory assistant.

    These will be executed in parallel.
    """
    configurable = configuration.Configuration.from_runnable_config(config)
    sends = []
    current_state = asdict(state)
    for v in configurable.memory_types:
        update_mode = v.update_mode
        match update_mode:
            case "patch":
                target = "handle_patch_memory"
            case "insert":
                target = "handle_insertion_memory"
            case _:
                raise ValueError(f"Unknown update mode: {update_mode}")

        sends.append(
            Send(
                target,
                ProcessorState(**{**current_state, "function_name": v.name}),
            )
        )
    return sends


builder.add_conditional_edges(
    "__start__", scatter_schemas, ["handle_patch_memory", "handle_insertion_memory"]
)

graph = builder.compile()


__all__ = ["graph"]
