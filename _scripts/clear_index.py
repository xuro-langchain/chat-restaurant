"""Clear Weaviate index using v4 client and current vector store setup."""

import logging
import os

from langchain_weaviate import WeaviateVectorStore
from langchain.indexes import SQLRecordManager, index
import weaviate

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.embeddings import get_embeddings_model

logger = logging.getLogger(__name__)


def clear() -> None:
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    embedding = get_embeddings_model()

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as client:
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )

        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager.create_schema()

        indexing_stats = index(
            [],
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
        )

        logger.info("Indexing stats: %s", indexing_stats)
        num_vecs = (
            client.collections.get(WEAVIATE_DOCS_INDEX_NAME)
            .aggregate.over_all()
            .total_count
        )
        logger.info("Vectors after clear: %s", num_vecs)


if __name__ == "__main__":
    clear()
