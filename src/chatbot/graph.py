"""Example chatbot that incorporates user memories."""

import datetime
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.graph import StateGraph
from langgraph.graph.message import Messages, add_messages
from langgraph_sdk import get_client
from typing_extensions import Annotated

from chatbot.configuration import ChatConfigurable
from chatbot.utils import format_memories


@dataclass
class ChatState:
    """The state of the chatbot."""

    messages: Annotated[list[Messages], add_messages]


llm = init_chat_model()


async def bot(state: ChatState) -> dict[str, list[Messages]]:
    """Prompt the bot to resopnd to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_context()
    namespace = ("memories", configurable.user_id,)
    store = get_store()
    # This lists ALL user memories in the provided namespace (up to the `limit`)
    # you can also filter by content.
    query = "\n".join(str(message.content) for message in state.messages)
    items = await store.asearch(namespace, query=query, limit=10)

    prompt = configurable.system_prompt.format(
        user_info=format_memories(items),
        time=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )
    m = await llm.ainvoke(
        [{"role": "system", "content": prompt}, *state.messages],
        config={"configurable": {"model": configurable.model}},
    )

    return {"messages": [m]}


async def schedule_memories(state: ChatState, config: RunnableConfig) -> None:
    """Prompt the bot to respond to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_context()
    memory_client = get_client()
    await memory_client.runs.create(
        # We enqueue the memory formation process on the same thread.
        # This means that IF this thread doesn't receive more messages before `after_seconds`,
        # it will read from the shared state and extract memories for us.
        # If a new request comes in for this thread before the scheduled run is executed,
        # that run will be canceled, and a **new** one will be scheduled once
        # this node is executed again.
        thread_id=config["configurable"]["thread_id"],
        # This memory-formation run will be enqueued and run later
        # If a new run comes in before it is scheduled, it will be cancelled,
        # then when this node is executed again, a *new* run will be scheduled
        multitask_strategy="enqueue",
        # This lets us "debounce" repeated requests to the memory graph
        # if the user is actively engaging in a conversation. This saves us $$ and
        # can help reduce the occurrence of duplicate memories.
        after_seconds=configurable.delay_seconds,
        # Specify the graph and/or graph configuration to handle the memory processing
        assistant_id=configurable.mem_assistant_id,
        input={"messages": state.messages},
        config={
            "configurable": {
                # Ensure the memory service knows where to save the extracted memories
                "user_id": configurable.user_id,
                "memory_types": configurable.memory_types,
            },
        },
    )


builder = StateGraph(ChatState, config_schema=ChatConfigurable)
builder.add_node(bot)
builder.add_node(schedule_memories)

builder.add_edge("__start__", "bot")
builder.add_edge("bot", "schedule_memories")

graph = builder.compile()
