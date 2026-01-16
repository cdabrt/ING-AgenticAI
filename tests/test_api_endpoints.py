import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_path = tmp_path / "artifacts" / "requirements.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("REQUIREMENTS_OUTPUT", str(output_path))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))
    monkeypatch.setenv("REQUIREMENTS_PDF_OUTPUT", str(tmp_path / "artifacts" / "requirements.pdf"))
    monkeypatch.setenv("DECISION_LOG_PATH", str(tmp_path / "artifacts" / "agent_decisions.jsonl"))

    import AgenticAI.main as app_module

    importlib.reload(app_module)
    client = TestClient(app_module.app)
    yield client, app_module, data_dir, output_path


def _make_pdf_bytes(text: str = "Hello world") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text=text, new_x="LMARGIN", new_y="NEXT")
    raw = pdf.output()
    return bytes(raw)


def test_root_endpoint(app_client):
    client, _, _, _ = app_client

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_pdfs_empty_when_no_files(app_client):
    client, _, _, _ = app_client

    response = client.get("/api/pdfs")
    assert response.status_code == 200
    assert response.json() == []


def test_upload_and_download_pdf(app_client):
    client, _, _, _ = app_client

    payload = {"file": ("sample.pdf", _make_pdf_bytes(), "application/pdf")}
    response = client.post("/api/upload", files=payload)
    assert response.status_code == 200

    list_response = client.get("/api/pdfs")
    assert list_response.status_code == 200
    data = list_response.json()
    assert len(data) == 1

    pdf_id = data[0]["id"]
    meta_response = client.get(f"/api/pdfs/{pdf_id}")
    assert meta_response.status_code == 200
    assert meta_response.json()["filename"] == "sample.pdf"

    download_response = client.get(f"/api/pdfs/{pdf_id}/download")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"].startswith("application/pdf")
    assert download_response.content.startswith(b"%PDF")


def test_upload_rejects_invalid_file(app_client):
    client, _, _, _ = app_client

    payload = {"file": ("note.txt", b"not a pdf", "text/plain")}
    response = client.post("/api/upload", files=payload)
    assert response.status_code == 400


def test_upload_rejects_duplicates(app_client):
    client, _, _, _ = app_client

    payload = {"file": ("dup.pdf", _make_pdf_bytes(), "application/pdf")}
    response = client.post("/api/upload", files=payload)
    assert response.status_code == 200

    duplicate = client.post("/api/upload", files=payload)
    assert duplicate.status_code == 409


def test_bundles_empty_when_no_output(app_client):
    client, _, _, _ = app_client

    response = client.get("/api/bundles")
    assert response.status_code == 200
    assert response.json() == []


def test_pipeline_endpoint_writes_output(app_client, monkeypatch):
    client, app_module, data_dir, output_path = app_client

    async def fake_run_pipeline(args, progress_callback=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "document": "Test",
                        "document_type": "Mock",
                        "business_requirements": [],
                        "data_requirements": [],
                        "assumptions": [],
                    }
                ],
                handle,
            )

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)

    response = client.post("/api/pipeline")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["document"] == "Test"
    assert output_path.exists()
