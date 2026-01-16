from contextlib import asynccontextmanager
from fastapi import FastAPI

from router import bundles, pdfs, requirements
from database.models.models import Base
from database.engine.psycopg_connection import create_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database tables
    engine = create_engine()
    # Drop and recreate tables to ensure schema is up to date
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized!")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.include_router(pdfs.router, prefix="/api")
app.include_router(bundles.router, prefix="/api")
app.include_router(requirements.router, prefix="/api")