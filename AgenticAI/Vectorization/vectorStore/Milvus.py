import json
import logging
import os
from typing import Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from AgenticAI.Vectorization.vectorStore.VectorStoreAdapter import IVectorStore

logger = logging.getLogger(__name__)


class MilvusStore(IVectorStore):
    def __init__(
        self,
        *,
        collection_name: str,
        dimensions: Optional[int],
        use_cosine_similarity: bool = True,
        host: Optional[str] = None,
        port: Optional[str] = None,
        uri: Optional[str] = None,
        auto_create: bool = False,
        connection_alias: str = "default",
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.uri = uri
        self.auto_create = auto_create
        self.connection_alias = connection_alias

        self._connect()
        collection = self._load_or_create_collection(dimensions, use_cosine_similarity)
        self.collection = collection

        embedding_dim = self._get_embedding_dimension(collection)
        super().__init__(embedding_dim, use_cosine_similarity)

    @classmethod
    def from_env(
        cls,
        *,
        dimensions: Optional[int],
        use_cosine_similarity: bool = True,
        collection_name: Optional[str] = None,
        auto_create: bool = False,
    ) -> "MilvusStore":
        return cls(
            collection_name=collection_name or os.getenv("MILVUS_COLLECTION", "agenticai_chunks"),
            dimensions=dimensions,
            use_cosine_similarity=use_cosine_similarity,
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT"),
            uri=os.getenv("MILVUS_URI"),
            auto_create=auto_create,
        )

    @classmethod
    def collection_exists(
        cls,
        *,
        collection_name: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        uri: Optional[str] = None,
        connection_alias: str = "default",
    ) -> bool:
        if not connections.has_connection(connection_alias):
            if uri:
                connections.connect(alias=connection_alias, uri=uri)
            else:
                connections.connect(
                    alias=connection_alias,
                    host=host or os.getenv("MILVUS_HOST", "milvus"),
                    port=port or os.getenv("MILVUS_PORT", "19530"),
                )
        return utility.has_collection(collection_name, using=connection_alias)

    def _connect(self) -> None:
        if connections.has_connection(self.connection_alias):
            return
        if self.uri:
            connections.connect(alias=self.connection_alias, uri=self.uri)
            return
        connections.connect(
            alias=self.connection_alias,
            host=self.host or os.getenv("MILVUS_HOST", "milvus"),
            port=self.port or os.getenv("MILVUS_PORT", "19530"),
        )

    def _load_or_create_collection(
        self,
        dimensions: Optional[int],
        use_cosine_similarity: bool,
    ) -> Collection:
        if utility.has_collection(self.collection_name, using=self.connection_alias):
            collection = Collection(self.collection_name, using=self.connection_alias)
            if dimensions is not None:
                existing_dim = self._get_embedding_dimension(collection)
                if existing_dim != dimensions:
                    raise ValueError(
                        f"Milvus collection {self.collection_name} has dimension {existing_dim}, expected {dimensions}"
                    )
            if not collection.indexes:
                metric_type = "IP" if use_cosine_similarity else "L2"
                index_params = {
                    "index_type": "HNSW",
                    "metric_type": metric_type,
                    "params": {"M": 16, "efConstruction": 200},
                }
                collection.create_index(field_name="embedding", index_params=index_params)
            return collection

        if not self.auto_create:
            raise ValueError(f"Milvus collection {self.collection_name} does not exist")

        if dimensions is None:
            raise ValueError("Dimensions are required to create a Milvus collection")

        fields = [
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=64,
            ),
            FieldSchema(
                name="doc_source",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="chunk_json",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dimensions,
            ),
        ]
        schema = CollectionSchema(fields, description="AgenticAI regulation chunks")
        collection = Collection(self.collection_name, schema, using=self.connection_alias)

        metric_type = "IP" if use_cosine_similarity else "L2"
        index_params = {
            "index_type": "HNSW",
            "metric_type": metric_type,
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("Created Milvus collection %s (dim=%s, metric=%s)", self.collection_name, dimensions, metric_type)
        return collection

    @staticmethod
    def _get_embedding_dimension(collection: Collection) -> int:
        for field in collection.schema.fields:
            if field.name == "embedding":
                return int(field.params.get("dim"))
        raise ValueError("Milvus collection schema missing embedding dimension")

    @staticmethod
    def _extract_source(chunk_json: str) -> str:
        try:
            payload = json.loads(chunk_json)
            return payload["document"]["meta_data"]["source"]
        except Exception:
            return ""

    def _ensure_loaded(self) -> None:
        self.collection.load()

    def store_embeds_and_metadata(self, chunks_with_embeds: List[Dict]):
        if not chunks_with_embeds:
            logger.warning("Received empty embedding batch; skipping store")
            return

        embeddings = [chunk["embedding"] for chunk in chunks_with_embeds]
        if len(embeddings[0]) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(embeddings[0])}"
            )

        chunk_ids = [chunk["chunk_id"] for chunk in chunks_with_embeds]
        chunk_jsons = [chunk["chunk"] for chunk in chunks_with_embeds]
        doc_sources = [self._extract_source(chunk_json) for chunk_json in chunk_jsons]

        self.collection.insert([chunk_ids, doc_sources, chunk_jsons, embeddings])
        self.collection.flush()
        logger.info("Inserted %s chunks into Milvus collection %s", len(chunk_ids), self.collection_name)

    def top_k_search(self, query_embedding, top_k=5):
        if len(query_embedding) != self.dimensions:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimensions}, got {len(query_embedding)}"
            )

        self._ensure_loaded()
        search_params = {"metric_type": "IP" if self.use_cosine_similarity else "L2", "params": {"ef": 64}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_json"],
        )
        payload = []
        for hit in results[0]:
            payload.append({"chunk": hit.entity.get("chunk_json"), "score": float(hit.score)})
        return payload

    def delete_by_sources(self, sources: List[str]) -> int:
        if not sources:
            return 0
        removed = 0
        for source in sources:
            expr = f"doc_source == {json.dumps(source)}"
            delete_result = self.collection.delete(expr)
            removed += getattr(delete_result, "delete_count", 0)
        self.collection.flush()
        logger.info("Deleted %s chunks from Milvus collection %s", removed, self.collection_name)
        return removed

    def count(self) -> int:
        return int(self.collection.num_entities)
