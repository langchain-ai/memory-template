"""Example chatbot that incorporates user memories."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.message import Messages, add_messages
from langgraph.store.base import BaseStore
from langgraph_sdk import get_client
from typing_extensions import Annotated

from chatbot.configuration import ChatConfigurable
from chatbot.utils import format_memories, init_model


@dataclass
class ChatState:
    """The state of the chatbot."""

    messages: Annotated[list[Messages], add_messages]


async def bot(
    state: ChatState, config: RunnableConfig, store: BaseStore
) -> dict[str, list[Messages]]:
    """Prompt the bot to resopnd to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_runnable_config(config)
    namespace = (configurable.user_id,)
    # This lists ALL user memories in the provided namespace (up to the `limit`)
    # you can also filter by content.
    items = await store.asearch(namespace)

    model = init_model(configurable.model)
    prompt = configurable.system_prompt.format(
        user_info=format_memories(items),
        time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    m = await model.ainvoke(
        [{"role": "system", "content": prompt}, *state.messages],
    )

    return {"messages": [m]}


async def schedule_memories(state: ChatState, config: RunnableConfig) -> None:
    """Prompt the bot to respond to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_runnable_config(config)
    memory_client = get_client()
    mem_thread_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            configurable.mem_assistant_id + config["configurable"]["thread_id"],
        )
    )
    await memory_client.threads.create(thread_id=mem_thread_id, if_exists="do_nothing")
    await memory_client.runs.create(
        # Generate a thread so we can run the memory service on a separate
        # but consistent thread. This lets us cancel scheduled runs if
        # a new message arrives to our chatbot before the memory service
        # begins processing.
        thread_id=mem_thread_id,
        # Rollback & cancel any scheduled runs for the target thread
        # that haven't completed
        multitask_strategy="rollback",
        # This lets us "debounce" repeated requests to the memory graph
        # if the user is actively engaging in a conversation
        after_seconds=configurable.delay_seconds,
        # Specify the graph and/or graph configuration to handle the memory processing
        assistant_id=configurable.mem_assistant_id,
        input={
            # the service dedupes messages by ID, so we can send the full convo each time
            # if we want
            "messages": state.messages,
        },
        config={
            "configurable": {
                # Ensure the memory service knows where to save the extracted memories
                "user_id": configurable.user_id,
            },
        },
    )


builder = StateGraph(ChatState, config_schema=ChatConfigurable)
builder.add_node(bot)
builder.add_node(schedule_memories)

builder.add_edge("__start__", "bot")
builder.add_edge("bot", "schedule_memories")

graph = builder.compile()
