"""
Initialize the database by creating all tables defined in the models.
"""
from contracts.models import Base
from database.engine.psycopg_connection import create_engine

def init_database(drop_existing=False):
    """Create all tables in the database.
    
    Args:
        drop_existing: If True, drops all existing tables before creating new ones
    """
    engine = create_engine()
    
    if drop_existing:
        print("Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)
    
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    # Set to True to drop and recreate all tables
    init_database(drop_existing=True)
