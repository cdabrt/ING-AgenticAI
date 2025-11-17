from fastapi import FastAPI

from AgenticAI.agentic.pipeline_runner import mock_run_pipeline

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/pipeline")
def run_pipeline():
    return mock_run_pipeline()