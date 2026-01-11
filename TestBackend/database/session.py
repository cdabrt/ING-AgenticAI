from sqlalchemy.orm import sessionmaker
from .engine.psycopg_connection import create_engine

engine = create_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)