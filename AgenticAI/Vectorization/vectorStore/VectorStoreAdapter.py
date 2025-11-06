from abc import abstractmethod, ABC
from typing import List, Dict


class IVectorStore(ABC):
    def __init__(self, dimensions:int, use_cosine_similarity=True):
        self.dimensions = dimensions
        self.use_cosine_similarity = use_cosine_similarity

    @abstractmethod
    def store_embeds_and_metadata(self, chunks_with_embeds : List[Dict]):
        pass

    @abstractmethod
    def search(self, query_embedding, top_k=5):
        pass