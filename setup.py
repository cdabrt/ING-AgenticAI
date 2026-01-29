from setuptools import setup, find_packages

setup(
    name="ING-AgenticAI",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pdfplumber",
        "sentence-transformers",
        "numpy",
        "pydantic",
        "faiss-cpu",
        "langchain",
        "langgraph",
        "langchain-google-genai",
        "langchain-openai",
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
