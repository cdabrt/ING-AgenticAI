# TODO: Setup docker container if necessary later

from setuptools import setup, find_packages

setup(
    name="ING-AgenticAI",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pdfplumber>=0.10.0",
        "sentence-transformers>=2.0.0",
        "numpy>=1.24.0",
        # Using the BaseModel superclass from pydantic to json serialize chunks and documents with ease with type validation.
        "pydantic>=2.0.0",
        "faiss-cpu>=1.7.0",
        "langchain>=0.1.0",
        "langgraph>=0.0.20",
        "langchain-google-genai>=1.0.0",
        "ddgs",
        "modelcontextprotocol",
        "python-dotenv>=1.0.0",
        "httpx>=0.25.0",
        "beautifulsoup4>=4.12.0",
        "fpdf2>=2.7.9",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0"
    ],
)
