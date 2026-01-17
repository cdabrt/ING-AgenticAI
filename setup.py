# TODO: Setup docker container if necessary later

from setuptools import setup, find_packages

setup(
    name="ING-AgenticAI",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pdfplumber",
        "sentence-transformers",
        "numpy",
        # Using the BaseModel superclass from pydantic to json serialize chunks and documents with ease with type validation.
        "pydantic",
        "faiss-cpu",
        "langchain",
        "langgraph",
        "langchain-google-genai",
        "langchain-openai",
        "pymilvus",
        "ddgs",
        "modelcontextprotocol",
        "python-dotenv",
        "httpx",
        "beautifulsoup4",
        "fpdf2",
        "fastapi",
        "uvicorn",
    ],
)
