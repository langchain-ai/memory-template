"""Define utility functions for your graph."""

from langgraph.store.base import Item


def format_memories(memories: list[Item] | None) -> str:
    """Format the user's memories."""
    if not memories:
        return ""
    # Note Bene: You can format better than this....
    formatted_memories = "\n".join(
        f"{str(m.value)}\tLast updated: {m.updated_at}" for m in memories
    )
    return f"""

## Memories

You have noted the following memorable events from previous interactions with the user.
<memories>
{formatted_memories}
</memories>
"""
