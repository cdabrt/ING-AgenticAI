import json
import logging
import os
from typing import Dict, List, Optional

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    LexicalHighlighter,
    RRFRanker,
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
        enable_hybrid: bool = False,
        connection_alias: str = "default",
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.uri = uri
        self.auto_create = auto_create
        self.connection_alias = connection_alias
        self.enable_hybrid = enable_hybrid
        self.text_field = "chunk_text"
        self.dense_field = "embedding"
        self.sparse_field = "sparse_embedding"

        self._connect()
        collection = self._load_or_create_collection(dimensions, use_cosine_similarity)
        self.collection = collection
        self.has_text_field = any(field.name == self.text_field for field in collection.schema.fields)
        self.has_sparse_field = any(field.name == self.sparse_field for field in collection.schema.fields)

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
        enable_hybrid: Optional[bool] = None,
    ) -> "MilvusStore":
        if enable_hybrid is None:
            enable_hybrid = os.getenv("MILVUS_ENABLE_HYBRID", "true").strip().lower() in ("1", "true", "yes", "y")
        return cls(
            collection_name=collection_name or os.getenv("MILVUS_COLLECTION", "agenticai_chunks"),
            dimensions=dimensions,
            use_cosine_similarity=use_cosine_similarity,
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT"),
            uri=os.getenv("MILVUS_URI"),
            auto_create=auto_create,
            enable_hybrid=enable_hybrid,
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
            if self.enable_hybrid:
                field_names = {field.name for field in collection.schema.fields}
                if self.text_field not in field_names or self.sparse_field not in field_names:
                    raise ValueError(
                        f"Milvus collection {self.collection_name} is missing hybrid fields. "
                        "Rebuild embeddings with MILVUS_ENABLE_HYBRID=true."
                    )
            if not collection.indexes:
                metric_type = "IP" if use_cosine_similarity else "L2"
                index_params = {
                    "index_type": "HNSW",
                    "metric_type": metric_type,
                    "params": {"M": 16, "efConstruction": 200},
                }
                collection.create_index(field_name=self.dense_field, index_params=index_params)
            if self.enable_hybrid:
                sparse_index = next((index for index in collection.indexes if index.field_name == self.sparse_field), None)
                if sparse_index is None:
                    sparse_params = {
                        "index_type": "SPARSE_INVERTED_INDEX",
                        "metric_type": "BM25",
                        "params": {"drop_ratio_build": 0.2},
                    }
                    collection.create_index(field_name=self.sparse_field, index_params=sparse_params)
                else:
                    metric = str(sparse_index.params.get("metric_type", "")).upper()
                    if metric and metric != "BM25":
                        raise ValueError(
                            f"Milvus collection {self.collection_name} has sparse index metric {metric}, expected BM25. "
                            "Recreate the collection with MILVUS_ENABLE_HYBRID=true."
                        )
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
        ]
        if self.enable_hybrid:
            fields.append(
                FieldSchema(
                    name=self.text_field,
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
            )
        fields.append(
            FieldSchema(
                name=self.dense_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=dimensions,
            )
        )
        functions = None
        if self.enable_hybrid:
            fields.append(
                FieldSchema(
                    name=self.sparse_field,
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                )
            )
            functions = [
                Function(
                    name="bm25_fn",
                    function_type=FunctionType.BM25,
                    input_field_names=[self.text_field],
                    output_field_names=[self.sparse_field],
                )
            ]
        schema = CollectionSchema(fields, description="AgenticAI regulation chunks", functions=functions)
        collection = Collection(self.collection_name, schema, using=self.connection_alias)

        metric_type = "IP" if use_cosine_similarity else "L2"
        index_params = {
            "index_type": "HNSW",
            "metric_type": metric_type,
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index(field_name=self.dense_field, index_params=index_params)
        if self.enable_hybrid:
            sparse_index = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {"drop_ratio_build": 0.2},
            }
            collection.create_index(field_name=self.sparse_field, index_params=sparse_index)
        logger.info(
            "Created Milvus collection %s (dim=%s, metric=%s, hybrid=%s)",
            self.collection_name,
            dimensions,
            metric_type,
            self.enable_hybrid,
        )
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

    @staticmethod
    def _extract_text(chunk_json: str) -> str:
        try:
            payload = json.loads(chunk_json)
            return payload["document"]["page_content"]
        except Exception:
            return ""

    @staticmethod
    def _extract_page(chunk_json: str) -> Optional[int]:
        try:
            payload = json.loads(chunk_json)
            return payload["document"]["meta_data"]["page"]
        except Exception:
            return None

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
        doc_sources = [chunk.get("doc_source") or self._extract_source(chunk_json) for chunk, chunk_json in zip(chunks_with_embeds, chunk_jsons)]
        chunk_texts = [
            chunk.get("chunk_text") or self._extract_text(chunk_json)
            for chunk, chunk_json in zip(chunks_with_embeds, chunk_jsons)
        ]

        fields = [chunk_ids, doc_sources, chunk_jsons]
        if self.has_text_field:
            fields.append(chunk_texts)
        fields.append(embeddings)
        self.collection.insert(fields)
        self.collection.flush()
        logger.info("Inserted %s chunks into Milvus collection %s", len(chunk_ids), self.collection_name)

    def top_k_search(self, query_embedding, top_k=5, query_text: str | None = None):
        if len(query_embedding) != self.dimensions:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimensions}, got {len(query_embedding)}"
            )

        self._ensure_loaded()
        if self.enable_hybrid and query_text:
            hybrid_limit = max(20, top_k * 4)
            dense_params = {"metric_type": "IP" if self.use_cosine_similarity else "L2", "params": {"ef": 64}}
            sparse_params = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}
            dense_req = AnnSearchRequest(
                data=[query_embedding],
                anns_field=self.dense_field,
                param=dense_params,
                limit=hybrid_limit,
            )
            sparse_req = AnnSearchRequest(
                data=[query_text],
                anns_field=self.sparse_field,
                param=sparse_params,
                limit=hybrid_limit,
            )
            try:
                results = self.collection.hybrid_search(
                    [dense_req, sparse_req],
                    rerank=RRFRanker(k=60),
                    limit=top_k,
                    output_fields=["chunk_json"],
                )
            except Exception as exc:
                logger.warning("Hybrid search failed; falling back to dense-only: %s", exc)
                results = self.collection.search(
                    data=[query_embedding],
                    anns_field=self.dense_field,
                    param=dense_params,
                    limit=top_k,
                    output_fields=["chunk_json"],
                )
        else:
            search_params = {"metric_type": "IP" if self.use_cosine_similarity else "L2", "params": {"ef": 64}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field=self.dense_field,
                param=search_params,
                limit=top_k,
                output_fields=["chunk_json"],
            )
        payload = []
        for hit in results[0]:
            payload.append({"chunk": hit.entity.get("chunk_json"), "score": float(hit.score)})
        return payload

    def highlight_search(
        self,
        query_text: str,
        *,
        chunk_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        if not query_text or not query_text.strip():
            return []
        if not self.has_text_field or not self.has_sparse_field:
            raise ValueError("Milvus collection does not support text highlighting")

        self._ensure_loaded()
        expr = None
        if chunk_ids:
            expr = f"chunk_id in {json.dumps(chunk_ids)}"

        highlighter = LexicalHighlighter(
            pre_tags=["[[H]]"],
            post_tags=["[[/H]]"],
            highlight_search_text=True,
            fragment_offset=12,
            fragment_size=220,
            num_of_fragments=2,
        )
        search_params = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}
        results = self.collection.search(
            data=[query_text],
            anns_field=self.sparse_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "chunk_json", self.text_field],
            highlighter=highlighter,
        )

        payload: List[Dict[str, object]] = []
        for hit in results[0]:
            highlight_block = None
            if hasattr(hit, "get"):
                highlight_block = hit.get("highlight")
            if highlight_block is None:
                highlight_block = getattr(hit, "highlight", None)
            highlight_block = highlight_block or {}
            fragments = highlight_block.get(self.text_field) or highlight_block.get("text") or []
            if not isinstance(fragments, list):
                fragments = [fragments]
            chunk_json = hit.entity.get("chunk_json")
            payload.append(
                {
                    "chunk_id": hit.entity.get("chunk_id"),
                    "source": self._extract_source(chunk_json) if chunk_json else None,
                    "page": self._extract_page(chunk_json) if chunk_json else None,
                    "highlights": fragments,
                    "score": float(hit.score),
                }
            )
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
