import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from fpdf import FPDF


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_pdf_bytes(text: str = "Hello world") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text=text, new_x="LMARGIN", new_y="NEXT")
    raw = pdf.output()
    return bytes(raw)


@pytest.fixture()
def e2e_server(tmp_path):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    vector_dir = tmp_path / "vector_store"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env["REQUIREMENTS_OUTPUT"] = str(artifacts_dir / "requirements.json")
    env["REQUIREMENTS_LATEST_OUTPUT"] = str(artifacts_dir / "requirements_latest.json")
    env["VECTOR_STORE_DIR"] = str(vector_dir)
    env["REQUIREMENTS_PDF_OUTPUT"] = str(artifacts_dir / "requirements.pdf")
    env["DECISION_LOG_PATH"] = str(artifacts_dir / "agent_decisions.jsonl")

    port = _get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "AgenticAI.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        ready = False
        for _ in range(50):
            if process.poll() is not None:
                stdout = process.stdout.read() if process.stdout else ""
                stderr = process.stderr.read() if process.stderr else ""
                pytest.fail(f"Uvicorn exited early. stdout={stdout} stderr={stderr}")
            try:
                response = httpx.get(f"{base_url}/", timeout=1.0)
                if response.status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                time.sleep(0.1)

        if not ready:
            stdout = process.stdout.read() if process.stdout else ""
            stderr = process.stderr.read() if process.stderr else ""
            pytest.fail(f"Uvicorn did not become ready. stdout={stdout} stderr={stderr}")

        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def test_e2e_upload_list_download(e2e_server):
    base_url = e2e_server
    payload = _make_pdf_bytes()

    with httpx.Client(timeout=5.0) as client:
        upload = client.post(
            f"{base_url}/api/upload",
            files={"file": ("sample.pdf", payload, "application/pdf")},
        )
        assert upload.status_code == 200

        listing = client.get(f"{base_url}/api/pdfs")
        assert listing.status_code == 200
        data = listing.json()
        assert len(data) == 1

        pdf_id = data[0]["id"]
        meta = client.get(f"{base_url}/api/pdfs/{pdf_id}")
        assert meta.status_code == 200
        assert meta.json()["filename"] == "sample.pdf"

        download = client.get(f"{base_url}/api/pdfs/{pdf_id}/download")
        assert download.status_code == 200
        assert download.headers["content-type"].startswith("application/pdf")
        assert download.content.startswith(b"%PDF")