"""Utility functions used in our graph."""

from typing import Sequence

from langchain_core.messages import AnyMessage, merge_message_runs


def prepare_messages(
    messages: Sequence[AnyMessage], system_prompt: str
) -> list[AnyMessage]:
    """Merge message runs and add instructions before and after to stay on task."""
    sys = {
        "role": "system",
        "content": f"""{system_prompt}

<memory-system>Reflect on following interaction. Use the provided tools to \
 retain any necessary memories about the user. Use parallel tool calling to handle updates & insertions simultaneously.</memory-system>
""",
    }
    m = {
        "role": "user",
        "content": "## End of conversation\n\n"
        "<memory-system>Reflect on the interaction above."
        " What memories ought to be retained or updated?</memory-system>",
    }
    return list(merge_message_runs(messages=[sys] + list(messages) + [m]))
