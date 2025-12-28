from abc import abstractmethod, ABC
from typing import List, Dict
from warnings import deprecated

from AgenticAI.Vectorization.StoredChunk import StoredChunk


class IVectorStore(ABC):
    def __init__(self, dimensions: int, use_cosine_similarity=True):
        self.dimensions = dimensions
        self.use_cosine_similarity = use_cosine_similarity

    @deprecated('do not used any longer')
    @abstractmethod
    def store_embeds_and_metadata(self, chunks_with_embeds: List[Dict]):
        pass

    @deprecated('do not used any longer')
    @abstractmethod
    def top_k_search(self, query_embedding, top_k=5):
        pass

    @abstractmethod
    def top_k_sparse_search(self, query: str, top_k=5) -> list[StoredChunk]:
        pass

    @abstractmethod
    def top_k_dense_search(self, query: str, top_k=5) -> list[StoredChunk]:
        pass

    @abstractmethod
    def top_k_hybrid_search(self, query: str, top_k=5, sparse_weight=0.5) -> list[StoredChunk]:
        pass

    @abstractmethod
    def sparse_search(self, query: str, rerank_score=0.35) -> list[StoredChunk]:
        pass

    @abstractmethod
    def dense_search(self, query: str, rerank_score=0.35) -> list[StoredChunk]:
        pass

    @abstractmethod
    def hybrid_search(self, query: str, sparse_weight=0.5, rerank_score=0.35) -> list[StoredChunk]:
        pass

    @abstractmethod
    def store_to_db(self, chunks: list[StoredChunk], batch_size=50) -> None:
        '''

        :param chunks: chunks without embeddings
        :param batch_size:
        :return:
        '''
        pass
