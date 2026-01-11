from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from AgenticAI.agentic.pipeline_runner import run_pipeline

app = FastAPI()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
VECTOR_DIR = os.getenv("VECTOR_STORE_DIR", "artifacts/vector_store")
OUTPUT_PATH = Path(os.getenv("REQUIREMENTS_OUTPUT", "artifacts/requirements.json"))
PDF_OUTPUT = os.getenv("REQUIREMENTS_PDF_OUTPUT", "artifacts/requirements.pdf")
DECISION_LOG = os.getenv("DECISION_LOG_PATH", "artifacts/agent_decisions.jsonl")
SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_servers" / "regulation_server.py")

PIPELINE_LOCK = asyncio.Lock()


def _list_pdfs() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.pdf"), key=lambda path: path.name.lower())


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


@app.post("/api/pipeline")
async def run_pipeline_endpoint():
    if PIPELINE_LOCK.locked():
        raise HTTPException(status_code=409, detail="Pipeline already running")

    if not DATA_DIR.exists():
        raise HTTPException(status_code=400, detail=f"Data directory not found: {DATA_DIR}")

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
        )
        await run_pipeline(args)

    return _load_requirement_bundles()


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
