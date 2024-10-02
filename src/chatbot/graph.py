"""Example chatbot that incorporates user memories."""

import os
import uuid
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages
from langgraph_sdk import get_client
from typing_extensions import Annotated

from chatbot.prompts import SYSTEM_PROMPT


@dataclass
class ChatState:
    """The state of the chatbot."""

    messages: Annotated[List[AnyMessage], add_messages]


@dataclass(kw_only=True)
class ChatConfigurable:
    """The configurable fields for the chatbot."""

    user_id: str
    mem_assistant_id: str
    model: str = "claude-3-5-sonnet-20240620"
    delay_seconds: int = 60  # For debouncing memory creation
    system_prompt: str = SYSTEM_PROMPT
    # None will default to connecting to the local deployment
    memory_service_url: str | None = None

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None):
        """Load configuration."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})


def format_memories(memories: Optional[list[dict]]) -> str:
    """Format the user's memories."""
    if not memories:
        return ""
    # Note Bene: You can format better than this....
    memories = "\n".join(str(m) for m in memories)
    return f"""

## Memories

You have noted the following memorable events from previous interactions with the user.
<memories>
{memories}
</memories>
"""


async def bot(state: ChatState, config: RunnableConfig) -> ChatState:
    """Prompt the bot to resopnd to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_runnable_config(config)
    memory_client = get_client(url=configurable.memory_service_url)
    namespace = (configurable.user_id,)
    # This lists ALL user memories in the provided namespace (up to the `limit`)
    # you can also filter by content.
    user_memory = await memory_client.store.search_items(namespace)

    model = init_chat_model(configurable.model)
    prompt = configurable.system_prompt.format(
        user_info=format_memories([item["value"] for item in user_memory["items"]]),
        time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    m = await model.ainvoke(
        [{"role": "system", "content": prompt}] + state.messages,
    )

    return {"messages": [m]}


async def schedule_memories(state: ChatState, config: RunnableConfig) -> ChatState:
    """Prompt the bot to resopnd to the user, incorporating memories (if provided)."""
    configurable = ChatConfigurable.from_runnable_config(config)
    memory_client = get_client(url=configurable.memory_service_url)
    memory_thread = uuid.uuid5(
        uuid.NAMESPACE_DNS,
        configurable.user_id + config["configurable"]["thread_id"],
    )
    await memory_client.threads.create(thread_id=memory_thread, if_exists="do_nothing")
    await memory_client.runs.create(
        memory_thread,
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
        # This let's us "debounce" repeated requests to the memory graph
        # if the user is actively engaging in a conversation
        after_seconds=configurable.delay_seconds,
    )


builder = StateGraph(ChatState, config_schema=ChatConfigurable)
builder.add_node(bot)
builder.add_node(schedule_memories)

builder.add_edge("__start__", "bot")
builder.add_edge("bot", "schedule_memories")

graph = builder.compile()
