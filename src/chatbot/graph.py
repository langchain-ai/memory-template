"""Example chatbot that incorporates user memories."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.message import Messages, add_messages
from langgraph_sdk import get_client
from typing_extensions import Annotated

from chatbot.configuration import ChatConfigurable
from chatbot.utils import format_memories, init_model


@dataclass
class ChatState:
    """The state of the chatbot."""

    messages: Annotated[list[Messages], add_messages]


async def bot(state: ChatState, config: RunnableConfig) -> dict[str, list[Messages]]:
    """Prompt the bot to resopnd to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_runnable_config(config)
    memory_client = get_client(url=configurable.memory_service_url)
    namespace = (configurable.user_id,)
    # This lists ALL user memories in the provided namespace (up to the `limit`)
    # you can also filter by content.
    user_memory = await memory_client.store.search_items(namespace)

    model = init_model(configurable.model)
    prompt = configurable.system_prompt.format(
        user_info=format_memories([item for item in user_memory["items"]]),
        time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    m = await model.ainvoke(
        [{"role": "system", "content": prompt}, *state.messages],
    )

    return {"messages": [m]}


async def schedule_memories(state: ChatState, config: RunnableConfig) -> None:
    """Prompt the bot to respond to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_runnable_config(config)
    memory_client = get_client(url=configurable.memory_service_url)
    memory_thread = str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            configurable.user_id + config["configurable"]["thread_id"],
        )
    )
    await memory_client.threads.create(thread_id=memory_thread, if_exists="do_nothing")
    await memory_client.runs.create(
        thread_id=memory_thread,
        assistant_id=configurable.mem_assistant_id,
        input={
            # the service dedupes messages by ID, so we can send the full convo each time
            # if we want
            "messages": state.messages,
        },
        config={
            "configurable": {
                "user_id": configurable.user_id,
            },
        },
        multitask_strategy="rollback",
        # This lets us "debounce" repeated requests to the memory graph
        # if the user is actively engaging in a conversation
        after_seconds=configurable.delay_seconds,
    )


builder = StateGraph(ChatState, config_schema=ChatConfigurable)
builder.add_node(bot)
builder.add_node(schedule_memories)

builder.add_edge("__start__", "bot")
builder.add_edge("bot", "schedule_memories")

graph = builder.compile()
