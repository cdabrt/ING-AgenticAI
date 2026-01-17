from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from AgenticAI.agentic.pipeline_runner import run_pipeline
from AgenticAI.pipeline.ingestion import ingest_documents, vector_store_exists
from AgenticAI.Vectorization.vectorStore.store_factory import load_store_config, resolve_backend

app = FastAPI()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
VECTOR_DIR = os.getenv("VECTOR_STORE_DIR", "artifacts/vector_store")
OUTPUT_PATH = Path(os.getenv("REQUIREMENTS_OUTPUT", "artifacts/requirements.json"))
PDF_OUTPUT = os.getenv("REQUIREMENTS_PDF_OUTPUT", "artifacts/requirements.pdf")
DECISION_LOG = os.getenv("DECISION_LOG_PATH", "artifacts/agent_decisions.jsonl")
SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_servers" / "regulation_server.py")

PIPELINE_LOCK = asyncio.Lock()

PIPELINE_STATUS: Dict[str, Optional[str] | float | int] = {
    "state": "idle",
    "stage": None,
    "message": None,
    "progress": 0.0,
    "eta_seconds": None,
    "retry_after_seconds": None,
    "started_at": None,
    "updated_at": None,
}

EMBEDDING_STATUS: Dict[str, Optional[str] | float] = {
    "state": "idle",
    "stage": None,
    "message": None,
    "progress": 0.0,
    "started_at": None,
    "updated_at": None,
}

PIPELINE_SETTINGS: Dict[str, bool] = {
    "throttle_enabled": True,
}

PIPELINE_RATE_EMA: Optional[float] = None
PIPELINE_LAST_PROGRESS: Optional[float] = None
PIPELINE_LAST_PROGRESS_AT: Optional[datetime] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_status(
    *,
    state: str,
    stage: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    started_at: Optional[str] = None,
    retry_after_seconds: Optional[int] = None,
) -> None:
    global PIPELINE_RATE_EMA
    global PIPELINE_LAST_PROGRESS
    global PIPELINE_LAST_PROGRESS_AT

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    if state == "running" and stage == "starting":
        PIPELINE_RATE_EMA = None
        PIPELINE_LAST_PROGRESS = progress if progress is not None else 0.0
        PIPELINE_LAST_PROGRESS_AT = now
        PIPELINE_STATUS["eta_seconds"] = None
    elif progress is not None:
        if PIPELINE_LAST_PROGRESS is not None and PIPELINE_LAST_PROGRESS_AT is not None:
            delta_time = (now - PIPELINE_LAST_PROGRESS_AT).total_seconds()
            delta_progress = progress - PIPELINE_LAST_PROGRESS
            if delta_time >= 0.5 and delta_progress >= 0:
                rate = delta_progress / delta_time
                alpha = 0.2
                PIPELINE_RATE_EMA = rate if PIPELINE_RATE_EMA is None else alpha * rate + (1 - alpha) * PIPELINE_RATE_EMA
        PIPELINE_LAST_PROGRESS = progress
        PIPELINE_LAST_PROGRESS_AT = now

    PIPELINE_STATUS["state"] = state
    PIPELINE_STATUS["stage"] = stage
    PIPELINE_STATUS["message"] = message
    if progress is not None:
        PIPELINE_STATUS["progress"] = progress
    if retry_after_seconds is not None:
        PIPELINE_STATUS["retry_after_seconds"] = retry_after_seconds
    elif state == "running":
        PIPELINE_STATUS["retry_after_seconds"] = None
    if started_at is not None:
        PIPELINE_STATUS["started_at"] = started_at
    PIPELINE_STATUS["updated_at"] = now_iso

    if state == "running" and PIPELINE_STATUS.get("started_at"):
        try:
            started = datetime.fromisoformat(str(PIPELINE_STATUS["started_at"]))
            elapsed = (now - started).total_seconds()
            effective_progress = PIPELINE_STATUS.get("progress", 0.0)
            if (
                PIPELINE_RATE_EMA
                and effective_progress >= 0.02
                and elapsed >= 3
            ):
                remaining = max(0.0, 1.0 - effective_progress)
                eta_estimate = remaining / PIPELINE_RATE_EMA
                if eta_estimate <= 0:
                    PIPELINE_STATUS["eta_seconds"] = 0
                else:
                    previous_eta = PIPELINE_STATUS.get("eta_seconds")
                    if previous_eta and previous_eta > 0:
                        eta_smoothed = previous_eta * 0.7 + eta_estimate * 0.3
                    else:
                        eta_smoothed = eta_estimate
                    PIPELINE_STATUS["eta_seconds"] = int(max(1, round(eta_smoothed)))
            else:
                PIPELINE_STATUS["eta_seconds"] = None
        except ValueError:
            PIPELINE_STATUS["eta_seconds"] = None
    elif state in {"completed", "error"}:
        PIPELINE_STATUS["eta_seconds"] = 0
        PIPELINE_RATE_EMA = None
        PIPELINE_LAST_PROGRESS = None
        PIPELINE_LAST_PROGRESS_AT = None


def _update_embedding_status(
    *,
    state: str,
    stage: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    started_at: Optional[str] = None,
) -> None:
    now_iso = _now_iso()
    EMBEDDING_STATUS["state"] = state
    EMBEDDING_STATUS["stage"] = stage
    EMBEDDING_STATUS["message"] = message
    if progress is not None:
        EMBEDDING_STATUS["progress"] = progress
    if started_at is not None:
        EMBEDDING_STATUS["started_at"] = started_at
    EMBEDDING_STATUS["updated_at"] = now_iso


def _extract_retry_delay(message: str) -> Optional[int]:
    match = re.search(r"retryDelay['\"]?: ['\"]?([0-9.]+)s", message)
    if match:
        return int(round(float(match.group(1))))
    match = re.search(r"retry in ([0-9.]+)s", message)
    if match:
        return int(round(float(match.group(1))))
    return None


def _summarize_pipeline_error(message: str) -> str:
    if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
        return message
    if "perday" in message.lower() or "per day" in message.lower():
        return "Quota exceeded for this model (daily limit). Try again later or switch API key."
    retry_delay = _extract_retry_delay(message)
    if retry_delay:
        return f"Quota exceeded. Retry after {retry_delay}s."
    return "Quota exceeded. Please wait and retry."


def _list_pdfs() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.pdf"), key=lambda path: path.name.lower())


def _load_document_index(persist_dir: str) -> Dict[str, Dict[str, str]]:
    config = load_store_config(persist_dir)
    index_path = config.get("document_index") if config else None
    if index_path:
        path = Path(index_path)
    else:
        path = Path(persist_dir) / "document_index.json"

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {item["source"]: item for item in payload if isinstance(item, dict) and "source" in item}
    return {}


def _save_document_index(persist_dir: str, index: Dict[str, Dict[str, str]]) -> None:
    path = Path(persist_dir) / "document_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)


def _save_store_config(persist_dir: str, config: Dict) -> None:
    path = Path(persist_dir) / "store_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def _load_requirement_bundles() -> List[dict]:
    if not OUTPUT_PATH.exists():
        return []
    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid requirements JSON: {exc}") from exc

    if isinstance(payload, list):
        return payload
    return [payload]


def _get_pdf_by_id(pdf_id: int) -> Path:
    pdfs = _list_pdfs()
    index = pdf_id - 1
    if index < 0 or index >= len(pdfs):
        raise HTTPException(status_code=404, detail="PDF not found")
    return pdfs[index]


@app.get("/")
def read_root():
    return {"status": "ok"}


class PipelinePayload(BaseModel):
    skip_ingestion: bool = False


@app.post("/api/pipeline")
async def run_pipeline_endpoint(payload: PipelinePayload | None = None):
    if PIPELINE_LOCK.locked():
        raise HTTPException(status_code=409, detail="Pipeline already running")

    if not DATA_DIR.exists():
        raise HTTPException(status_code=400, detail=f"Data directory not found: {DATA_DIR}")

    if payload and payload.skip_ingestion:
        if not vector_store_exists(VECTOR_DIR):
            raise HTTPException(status_code=400, detail="Vector store not found. Run embeddings first.")

    _update_status(
        state="running",
        stage="starting",
        message="Initializing pipeline",
        progress=0.0,
        started_at=_now_iso(),
    )

    async with PIPELINE_LOCK:
        args = SimpleNamespace(
            data_dir=str(DATA_DIR),
            vector_dir=VECTOR_DIR,
            server_script=SERVER_SCRIPT,
            rebuild_store=False,
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "15")),
            output=str(OUTPUT_PATH),
            pdf_output=PDF_OUTPUT,
            decision_log=DECISION_LOG,
            throttle_enabled=PIPELINE_SETTINGS["throttle_enabled"],
            skip_ingestion=bool(payload.skip_ingestion) if payload else False,
        )

        def progress_callback(stage: str, progress: float, message: Optional[str] = None) -> None:
            _update_status(state="running", stage=stage, message=message, progress=progress)

        try:
            await run_pipeline(args, progress_callback=progress_callback)
        except Exception as exc:
            retry_delay = _extract_retry_delay(str(exc))
            _update_status(
                state="error",
                stage="error",
                message=_summarize_pipeline_error(str(exc)),
                progress=PIPELINE_STATUS["progress"],
                retry_after_seconds=retry_delay,
            )
            raise

    _update_status(state="completed", stage="complete", message="Pipeline finished", progress=1.0)

    return _load_requirement_bundles()


@app.get("/api/pipeline/status")
def get_pipeline_status():
    return PIPELINE_STATUS


@app.get("/api/embeddings/status")
def get_embedding_status():
    return EMBEDDING_STATUS


@app.get("/api/embeddings/index")
def get_embedding_index():
    pdfs = _list_pdfs()
    document_index = _load_document_index(VECTOR_DIR)
    results = []
    embedded_count = 0
    for idx, path in enumerate(pdfs, start=1):
        entry = document_index.get(path.name)
        embedded = entry is not None
        if embedded:
            embedded_count += 1
        results.append(
            {
                "id": idx,
                "filename": path.name,
                "size": path.stat().st_size,
                "embedded": embedded,
                "chunk_count": entry.get("chunk_count") if entry else None,
                "embedded_at": entry.get("updated_at") if entry else None,
            }
        )
    return {
        "pdfs": results,
        "embedded_count": embedded_count,
        "total": len(pdfs),
    }


@app.get("/api/vector-store")
def get_vector_store_status():
    config = load_store_config(VECTOR_DIR)
    if not config:
        return {"exists": False}

    backend = resolve_backend(config.get("vector_store"))
    payload = {
        "exists": True,
        "backend": backend,
        "config": config,
    }

    if backend == "milvus":
        collection_name = config.get("collection_name") or os.getenv("MILVUS_COLLECTION", "agenticai_chunks")
        try:
            from AgenticAI.Vectorization.vectorStore.Milvus import MilvusStore
        except ModuleNotFoundError:
            payload["collection_exists"] = False
            return payload

        collection_exists = MilvusStore.collection_exists(collection_name=collection_name)
        payload["collection_exists"] = collection_exists
        if collection_exists:
            store = MilvusStore.from_env(
                dimensions=config.get("dimension"),
                use_cosine_similarity=config.get("use_cosine_similarity", True),
                collection_name=collection_name,
                auto_create=False,
            )
            payload["collection_count"] = store.count()
        return payload

    payload["collection_count"] = config.get("chunk_count")
    return payload


class EmbeddingPayload(BaseModel):
    pdf_ids: Optional[List[int]] = None


@app.post("/api/embeddings")
async def run_embedding(payload: EmbeddingPayload | None = None):
    if PIPELINE_LOCK.locked():
        raise HTTPException(status_code=409, detail="Pipeline already running")

    if not DATA_DIR.exists():
        raise HTTPException(status_code=400, detail=f"Data directory not found: {DATA_DIR}")

    include_sources = None
    if payload and payload.pdf_ids:
        include_sources = []
        for pdf_id in payload.pdf_ids:
            path = _get_pdf_by_id(pdf_id)
            include_sources.append(path.name)

    _update_embedding_status(
        state="running",
        stage="starting",
        message="Preparing embeddings",
        progress=0.0,
        started_at=_now_iso(),
    )

    async with PIPELINE_LOCK:
        def progress_callback(progress: float, message: Optional[str] = None) -> None:
            _update_embedding_status(state="running", stage="embedding", message=message, progress=progress)

        try:
            result = await asyncio.to_thread(
                ingest_documents,
                data_dir=str(DATA_DIR),
                persist_dir=VECTOR_DIR,
                vector_store_backend=os.getenv("VECTOR_STORE_BACKEND"),
                include_sources=include_sources,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            _update_embedding_status(state="error", stage="error", message=str(exc), progress=EMBEDDING_STATUS["progress"])
            raise

    _update_embedding_status(state="completed", stage="complete", message="Embedding finished", progress=1.0)
    return result


@app.delete("/api/embeddings/{pdf_id}")
def delete_embedding(pdf_id: int):
    config = load_store_config(VECTOR_DIR)
    if not config:
        raise HTTPException(status_code=404, detail="Vector store not found")

    backend = resolve_backend(config.get("vector_store"))
    if backend != "milvus":
        raise HTTPException(status_code=400, detail="Embedding removal is only supported with Milvus.")

    path = _get_pdf_by_id(pdf_id)
    source = path.name
    document_index = _load_document_index(VECTOR_DIR)
    if source not in document_index:
        raise HTTPException(status_code=404, detail="PDF not embedded")

    collection_name = config.get("collection_name") or os.getenv("MILVUS_COLLECTION", "agenticai_chunks")
    try:
        from AgenticAI.Vectorization.vectorStore.Milvus import MilvusStore
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=500, detail="pymilvus is required for Milvus operations") from exc

    store = MilvusStore.from_env(
        dimensions=config.get("dimension"),
        use_cosine_similarity=config.get("use_cosine_similarity", True),
        collection_name=collection_name,
        auto_create=False,
    )
    deleted = store.delete_by_sources([source])

    document_index.pop(source, None)
    _save_document_index(VECTOR_DIR, document_index)

    config["chunk_count"] = store.count()
    config["generated_at"] = _now_iso()
    config["document_index"] = str((Path(VECTOR_DIR) / "document_index.json").resolve())
    _save_store_config(VECTOR_DIR, config)

    return {
        "source": source,
        "deleted_chunks": deleted,
        "remaining_chunks": config["chunk_count"],
    }


class ThrottlePayload(BaseModel):
    enabled: bool


@app.get("/api/pipeline/throttle")
def get_pipeline_throttle():
    return {"enabled": PIPELINE_SETTINGS["throttle_enabled"]}


@app.post("/api/pipeline/throttle")
def set_pipeline_throttle(payload: ThrottlePayload):
    PIPELINE_SETTINGS["throttle_enabled"] = payload.enabled
    return {"enabled": PIPELINE_SETTINGS["throttle_enabled"]}


@app.get("/api/bundles")
def get_requirement_bundles():
    return _load_requirement_bundles()


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / file.filename
    if target.exists():
        raise HTTPException(status_code=409, detail=f"A PDF named '{file.filename}' already exists")

    contents = await file.read()
    target.write_bytes(contents)
    return {"filename": file.filename, "size": len(contents)}


@app.get("/api/pdfs")
def get_pdfs():
    pdfs = _list_pdfs()
    results = []
    for idx, path in enumerate(pdfs, start=1):
        results.append(
            {
                "id": idx,
                "filename": path.name,
                "size": path.stat().st_size,
            }
        )
    return results


@app.get("/api/pdfs/{pdf_id}")
def get_pdf(pdf_id: int):
    path = _get_pdf_by_id(pdf_id)
    return {
        "id": pdf_id,
        "filename": path.name,
        "size": path.stat().st_size,
    }


@app.get("/api/pdfs/{pdf_id}/download")
def download_pdf(pdf_id: int):
    path = _get_pdf_by_id(pdf_id)
    return Response(
        content=path.read_bytes(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )
