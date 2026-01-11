from sqlalchemy import Engine, create_engine as sqlalchemy_create_engine

def create_engine() -> Engine:
    # Database connection string from docker-compose.local.test.yml
    return sqlalchemy_create_engine(
        "postgresql+psycopg2://postgres:Pass_word@db:5432/RequirementsDB",
        isolation_level="SERIALIZABLE",
    )