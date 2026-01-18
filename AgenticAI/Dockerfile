FROM python:3.11-slim

WORKDIR /app

COPY setup.py .
RUN pip install --no-cache-dir -e .

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "AgenticAI.main:app", "--host", "0.0.0.0", "--port", "8000"]
