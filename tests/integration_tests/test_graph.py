import uuid
from datetime import datetime
from typing import Literal, Optional

import langsmith as ls
import pytest
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from memory_graph.graph import builder


class User(BaseModel):
    """Store all important information about a user here."""

    preferred_name: Optional[str] = None
    current_age: Optional[str] = None
    skills: list[str] = Field(description="Various skills the user has.")
    favorite_foods: Optional[list[str]]
    last_updated: datetime
    core_memories: list[str] = Field(
        description="Important events and memories that shape the user's identity."
    )
    topics_discussed: list[str] = Field(
        description="topics the user has discussed previously"
    )
    other_preferences: list[str] = Field(
        description="Other preferences the user has expressed that informs how you should interact with them."
    )
    relationships: list[str] = Field(
        description="Store information about friends, family members, coworkers, and other important relationships the user has here. Include relevant information about htem."
    )


def create_function(model, description: str = ""):
    return {}


def create_memory_function(
    model,
    description: str = "",
    custom_instructions: str = "",
    kind: Literal["patch", "insert"] = "patch",
):
    return {
        "name": model.__name__,
        "description": description or model.__doc__ or "",
        "parameters": model.model_json_schema(),
        "system_prompt": custom_instructions,
        "update_mode": kind,
    }


@pytest.mark.asyncio
@ls.unit
async def test_patch_memory_stored():
    mem_store = InMemoryStore()
    mem_func = create_memory_function(User)
    graph = builder.compile(store=mem_store)
    thread_id = str(uuid.uuid4())
    user_id = "my-test-user"
    config = {
        "configurable": {"memory_types": [mem_func]},
        "thread_id": thread_id,
        "user_id": user_id,
    }
    await graph.ainvoke(
        {"messages": [("user", "My name is Bob. I like fun things")]}, config
    )
    namespace = (user_id, "user_states", "User")
    memories = mem_store.search(namespace)
    ls.expect(len(memories)).to_equal(1)
    mem = memories[0]
    ls.expect(mem.value.get("preferred_name")).to_equal("Bob")

    await graph.ainvoke(
        {
            "messages": [
                ("user", "Even though my name is Bob, I prefer to go by Robert.")
            ]
        },
        config,
    )
    memories = mem_store.search(namespace)
    ls.expect(len(memories)).to_equal(1)
    mem = memories[0]
    ls.expect(mem.value.get("preferred_name")).to_equal("Robert")

    # Check that searching by a different namespace returns no memories
    bad_namespace = ("user_states", "my-bad-test-user", "User")
    memories = mem_store.search(bad_namespace)
    ls.expect(memories).against(lambda x: not x)


class Relationship(BaseModel):
    """A relationship memory type for insertion.

    Call for each distinct individual the user interacts with. Don't forget to document each new relationship with a new entry.
    """

    name: str = Field(description="The legal name of the person.")
    preferred_name: str = Field(
        description="The name of the person in the relationship"
    )
    relation_to_user: str = Field(
        description="The type of relationship "
        "(e.g., friend, sister, brother, grandmother, colleague)"
    )
    recent_interactions: list[str] = Field(
        description="List of recent interactions with this person"
    )
    notes: str = Field(
        description="Other important information about this individual and how they relate to the user."
    )


@pytest.mark.asyncio
@ls.unit
async def test_insertion_memory_stored():
    mem_store = InMemoryStore()
    mem_func = create_memory_function(
        Relationship,
        custom_instructions="Extract all relationships mentioned. Call Relationship once per-relationship."
        " Use parallel tool calling to handle updates & insertions simultaneously.",
        kind="insert",
    )
    graph = builder.compile(store=mem_store)
    thread_id = str(uuid.uuid4())
    user_id = "my-test-user"
    config = {
        "configurable": {"memory_types": [mem_func]},
        "user_id": user_id,
    }
    await graph.ainvoke(
        {
            "messages": [
                (
                    "user",
                    "I've been thinking about my old friend Joanne Steine lately. We met in 3rd grade and were inseparable for years.",
                ),
                (
                    "assistant",
                    "It's nice that you're reminiscing about your childhood friend Joanne. Friendships from that age can be very special. How are things between you two now?",
                ),
                (
                    "user",
                    "Well, that's the thing. We've been drifting apart over the years. It's sad, but I guess it happens sometimes.",
                ),
                (
                    "assistant",
                    "I'm sorry to hear that you and Joanne have been drifting apart. It's true that relationships can change over time. Is there anything specific that's contributed to the distance between you?",
                ),
                (
                    "user",
                    "Not really, just life getting in the way I suppose. But you know, it makes me even more grateful for my friend Anthony. Despite living far apart, we still manage to stay connected.",
                ),
                (
                    "assistant",
                    "It's wonderful that you have a friend like Anthony who you can maintain a strong connection with despite the distance. Long-lasting friendships like that are truly valuable. How do you and Anthony manage to stay close?",
                ),
                (
                    "user",
                    "We make an effort to call each other regularly and share important moments in our lives. It's not always easy, but it's worth it.",
                ),
            ]
        },
        {**config, "thread_id": thread_id},
    )
    namespace = (user_id, "events", "Relationship")
    memories = mem_store.search(namespace)
    ls.expect(len(memories)).to_be_greater_than(1)
    # Check for Joanne's relationship
    joanne_relationship = next(
        (mem for mem in memories if "Joanne" in mem.value.get("name", "")), None
    )
    ls.expect(joanne_relationship).against(lambda x: x is not None)
    ls.expect(joanne_relationship.value.get("relation_to_user")).to_contain("friend")

    # Check for Anthony's relationship
    anthony_relationship = next(
        (mem for mem in memories if "Anthony" in mem.value.get("name")), None
    )
    ls.expect(anthony_relationship).against(lambda x: x is not None)
    ls.expect(anthony_relationship.value.get("relation_to_user")).to_contain("friend")

    thread_id_2 = str(uuid.uuid4())

    # New conversation about Joanne's preferred name
    await graph.ainvoke(
        {
            "messages": [
                (
                    "user",
                    "I just talked with Joanne. She told me she's going by 'Jo' now.",
                ),
                (
                    "assistant",
                    "Oh, that's interesting! It's nice that Joanne - or should I say Jo - shared that with you. How do you feel about her name change?",
                ),
                (
                    "user",
                    "I think it suits her. It's a bit strange to get used to, but I'm happy for her."
                    " She also introduced me to a great person named Nick."
                    " Nick and i became fast friends. I'm going surfing with him tomorrow.",
                ),
            ]
        },
        {**config, "thread_id": thread_id_2},
    )

    # Check the memories again
    updated_memories = mem_store.search(
        namespace,
    )
    ls.expect(len(updated_memories)).to_equal(
        3
    )  # Now there should be 3 objects: Nick, Joanne, and Anthony

    # Check for updated Joanne/Jo relationship
    jo_relationship = next(
        (mem for mem in updated_memories if "Joanne" in mem.value.get("name")), None
    )
    ls.expect(jo_relationship).against(lambda x: x is not None)
    ls.expect(jo_relationship.value.get("preferred_name")).to_equal("Jo")

    # Check for Nick's relationship
    nick_relationship = next(
        (mem for mem in updated_memories if "Nick" in mem.value.get("name", "")), None
    )
    ls.expect(nick_relationship).against(lambda x: x is not None)
    ls.expect(nick_relationship.value.get("relation_to_user")).to_contain("friend")
