import logging
from typing import override
from warnings import deprecated
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from sentence_transformers import CrossEncoder
from Vectorization.StoredChunk import StoredChunk
from Vectorization.vectorStore.VectorStoreAdapter import IVectorStore

logger = logging.getLogger(__name__)


class MilvusStore(IVectorStore):
    def __init__(self, uri: str, token: str, collection_name: str):
        logger.critical('have only tried the notebook, this is not tried')
        self.uri = uri
        self.token = token
        self.collection_name = collection_name

        self._create_models()
        if not self._connet_db():
            self._create_db()
        logger.info(f'{self.__class__.__name__} initialization finished')

    def __del__(self):
        connections.disconnect(self.collection_name)
        logger.info('db disconnected')

    def _connet_db(self) -> bool:
        connections.connect(uri=self.uri, token=self.token)
        if not utility.has_collection(self.collection_name):
            return False

        self.collection = Collection(self.collection_name)
        self.collection.load()
        return True

    def _create_db(self):
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=10240),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page", dtype=DataType.INT32),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="char_start", dtype=DataType.INT32),
            FieldSchema(name="char_end", dtype=DataType.INT32),
            FieldSchema(name="parent_heading", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_model.dim["dense"]),
        ]
        schema = CollectionSchema(fields)
        self.collection = Collection(self.collection_name, schema, consistency_level="Bounded")
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        self.collection.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        self.collection.create_index("dense_vector", dense_index)
        self.collection.load()

    def _create_models(self):
        # these tow can be replaced with any kinds of combinations, embedding model can be separated into sparse one and dense one
        self.embedding_model = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

    @override
    def store_embeds_and_metadata(self, chunks_with_embeds: list[StoredChunk]):
        raise NotImplementedError()

    @override
    def top_k_search(self, query_embedding, top_k=5):
        raise NotImplementedError()

    def _map_stored_chunk(self, record: dict[str, object]) -> StoredChunk:
        return StoredChunk.model_validate(record)

    def _rerank(self, query: str, results: list[StoredChunk]):
        pairs = [[query, result.page_content] for result in results]
        scores = self.rerank_model.predict(pairs)
        for r, s in zip(results, scores):
            r.score = s
        reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        return reranked_results

    def _query_embedding(self, query):
        query_embeddings = self.embedding_model([query])
        return query_embeddings["sparse"][[0]], query_embeddings["dense"][0]

    def _search_params(self):
        return {"metric_type": "IP", "params": {}}

    def _output_fields(self):
        return ['chunk_id', 'page_content', 'source', 'page', 'type', 'char_start', 'char_end', 'parent_heading']

    def _post_search(self, query, result, top_k):
        chunks = [self._map_stored_chunk(record['entity']) for record in result]
        chunks = self._rerank(query, chunks)
        return chunks[:top_k]

    def _top_k_search(self, query: str, field: str, top_k=20) -> list[StoredChunk]:
        query_sparse_embedding, query_dense_embedding = self._query_embedding(query)
        result = self.collection.search(
            [query_sparse_embedding if field == 'sparse_vector' else query_dense_embedding],
            anns_field=field,
            limit=2 * top_k,
            output_fields=self._output_fields(),
            param=self._search_params(),
        )[0]
        return self._post_search(query, result, top_k)

    @override
    def top_k_sparse_search(self, query: str, top_k=20) -> list[StoredChunk]:
        # query_sparse_embedding, _ = self._query_embedding(query)
        # result = self.collection.search(
        #     [query_sparse_embedding],
        #     anns_field="sparse_vector",
        #     limit=2 * top_k,
        #     output_fields=self._output_fields(),
        #     param=self._search_params(),
        # )[0]
        # return self._post_search(query, result, top_k)
        return self._top_k_search(query, 'sparse_vector', top_k)

    @override
    def top_k_dense_search(self, query, top_k=20) -> list[StoredChunk]:
        # _, query_dense_embedding = self._query_embedding(query)
        # result = self.collection.search(
        #     [query_dense_embedding],
        #     anns_field="dense_vector",
        #     limit=2 * top_k,
        #     output_fields=self._output_fields(),
        #     param=self._search_params(),
        # )[0]
        # return self._post_search(query, result, top_k)
        return self._top_k_search(query, 'dense_vector', top_k)

    @override
    def top_k_hybrid_search(self, query, top_k=20, sparse_weight=0.7) -> list[StoredChunk]:
        query_sparse_embedding, query_dense_embedding = self._query_embedding(query)
        dense_req = AnnSearchRequest(
            [query_dense_embedding],
            "dense_vector",
            self._search_params(),
            limit=2 * top_k
        )
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding],
            "sparse_vector",
            self._search_params(),
            limit=2 * top_k
        )
        reranker = WeightedRanker(sparse_weight, 1 - sparse_weight)
        result = self.collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=reranker,
            limit=2 * top_k,
            output_fields=self._output_fields()
        )[0]
        return self._post_search(query, result, top_k)

    @override
    def store_to_db(self, chunks: list[StoredChunk], batch_size=50):
        length = len(chunks)
        for i in range(0, length, batch_size):
            batch = chunks[i:i + batch_size]
            content_list = [chunk.page_content for chunk in batch]
            docs_embeddings = self.embedding_model(content_list)
            batched_entities = [
                [chunk.chunk_id for chunk in batch],
                content_list,
                [chunk.source for chunk in batch],
                [chunk.page for chunk in batch],
                [chunk.type for chunk in batch],
                [chunk.char_start for chunk in batch],
                [chunk.char_end for chunk in batch],
                [chunk.parent_heading for chunk in batch],
                docs_embeddings["sparse"],
                docs_embeddings["dense"],
            ]
            self.collection.insert(batched_entities)

    def _search(self, embedding, query: str, field: str, page_number=0, length=10) -> list[StoredChunk]:
        '''

        :param query:
        :param page_number: starts from 0
        :param length:
        :return:
        '''
        result = self.collection.search(
            [embedding],
            anns_field=field,
            limit=length,
            output_fields=self._output_fields(),
            param={
                **self._search_params(),
                "offset": page_number * length
            }
        )[0]
        return self._post_search(query, result, length)

    def _sparse_search(self, query_sparse_embedding, query: str, page_number=0, length=10) -> list[StoredChunk]:
        return self._search(query_sparse_embedding, query, 'sparse_vector', page_number, length)

    def _continue_search(self, query: str, category: str, rerank_score=0.35, reranker=None) -> list[StoredChunk]:
        page_number = 0
        result = []
        query_sparse_embedding, query_dense_embedding = self._query_embedding(query)
        while True:
            if category == 'sparse':
                page_result = self._sparse_search(query_sparse_embedding, query, page_number)
            elif category == 'dense':
                page_result = self._dense_search(query_dense_embedding, query, page_number)
            elif category == 'hybrid':
                page_result = self._hybrid_search(query_sparse_embedding, query_dense_embedding, reranker, query,
                                                  page_number)
            else:
                raise RuntimeError('invalid category')

            if page_result[-1].score >= rerank_score:
                result += page_result
                continue
            else:
                for index, record in enumerate(page_result):
                    if record.score < rerank_score:
                        result += page_result[:index]
                        break
                break
        return sorted(result, key=lambda x: x.score, reverse=True)

    @override
    def sparse_search(self, query: str, rerank_score=0.35) -> list[StoredChunk]:
        return self._continue_search(query, 'sparse', rerank_score)

    def _dense_search(self, query_dense_embedding, query: str, page_number=0, length=10):
        return self._search(query_dense_embedding, query, 'dense_vector', page_number, length)

    @override
    def dense_search(self, query: str, rerank_score=0.35) -> list[StoredChunk]:
        return self._continue_search(query, 'dense', rerank_score)

    def _hybrid_search(self, sparse_embedding, dense_embedding, reranker, query: str, page_number=0, length=10) -> list[
        StoredChunk]:
        dense_req = AnnSearchRequest(
            [sparse_embedding],
            "dense_vector",
            self._search_params(),
            limit=length
        )
        sparse_req = AnnSearchRequest(
            [dense_embedding],
            "sparse_vector",
            self._search_params(),
            limit=length
        )
        result = self.collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=reranker,
            limit=length,
            output_fields=self._output_fields(),
            param={
                **self._search_params(),
                "offset": page_number * length
            }
        )[0]
        return self._post_search(query, result, length)

    @override
    def hybrid_search(self, query: str, sparse_weight=0.5, rerank_score=0.35) -> list[StoredChunk]:
        reranker = WeightedRanker(sparse_weight, 1 - sparse_weight)
        return self._continue_search(query, 'hybrid', rerank_score, reranker)
