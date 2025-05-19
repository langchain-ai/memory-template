import contextlib
from typing import cast

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langgraph.store.base import IndexConfig
from langgraph.store.sqlite import AsyncSqliteStore

embeddings = cast(Embeddings, init_embeddings("openai:text-embedding-3-small"))


@contextlib.asynccontextmanager
async def generate_store():
    """Generate a store, to be open for the duration of the server."""
    async with AsyncSqliteStore.from_conn_string(
        "./custom_store.sql",
        index=IndexConfig(
            dims=1536,
            embed=embeddings,
            fields=["$"],
        ),
    ) as store:
        await store.setup()
        yield store
