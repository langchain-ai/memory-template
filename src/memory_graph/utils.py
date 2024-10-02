"""Utility functions used in our graph."""

from typing import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, merge_message_runs


def prepare_messages(
    messages: Sequence[AnyMessage], system_prompt: str
) -> list[AnyMessage]:
    """Merge message runs and add instructions before and after to stay on task."""
    sys = {
        "role": "system",
        "content": f"""{system_prompt}

<memory-system>Reflect on following interaction. Use the provided tools to \
 retain any necessary memories about the user.</memory-system>
""",
    }
    m = {
        "role": "user",
        "content": "## End of conversation\n\n"
        "<memory-system>Reflect on the interaction above."
        " What memories ought to be retained or updated?</memory-system>",
    }
    return merge_message_runs([sys] + list(messages) + [m])


def init_model(fully_specified_name: str) -> BaseChatModel:
    """Initialize the configured chat model."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)