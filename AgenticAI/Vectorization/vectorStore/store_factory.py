import json
from pathlib import Path
from typing import Any, Dict, Optional

from AgenticAI.Vectorization.vectorStore.FAISS import FAISSStore


def resolve_backend(value: Optional[str]) -> str:
    if not value:
        return "faiss"
    return value.strip().lower() or "faiss"


def load_store_config(persist_dir: str) -> Dict[str, Any]:
    config_path = Path(persist_dir) / "store_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_vector_store(persist_dir: str, config: Dict[str, Any]):
    backend = resolve_backend(config.get("vector_store"))
    use_cosine_similarity = config.get("use_cosine_similarity", True)
    if backend == "milvus":
        try:
            from AgenticAI.Vectorization.vectorStore.Milvus import MilvusStore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pymilvus is required for VECTOR_STORE_BACKEND=milvus. "
                "Install it with `pip install pymilvus`."
            ) from exc
        return MilvusStore.from_env(
            dimensions=config.get("dimension"),
            use_cosine_similarity=use_cosine_similarity,
            collection_name=config.get("collection_name"),
            auto_create=False,
        )
    return FAISSStore.load(persist_dir, use_cosine_similarity=use_cosine_similarity)
