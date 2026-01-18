import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from AgenticAI.Vectorization.vectorStore.FAISS import FAISSStore


def resolve_backend(value: Optional[str]) -> str:
    if not value:
        return "faiss"
    return value.strip().lower() or "faiss"


def resolve_enable_hybrid(config: Dict[str, Any]) -> bool:
    env_value = os.getenv("MILVUS_ENABLE_HYBRID")
    if env_value is not None:
        return env_value.strip().lower() in ("1", "true", "yes", "y")
    if "enable_hybrid" in config:
        return bool(config["enable_hybrid"])
    return True


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
        enable_hybrid = resolve_enable_hybrid(config)
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
            enable_hybrid=enable_hybrid,
        )
    return FAISSStore.load(persist_dir, use_cosine_similarity=use_cosine_similarity)
