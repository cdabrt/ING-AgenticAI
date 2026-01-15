from TestBackend.repository.postgre_client import PostgreClient, get_db
from fastapi import Depends

class DatabaseClient:
    def __init__(self, db_client: PostgreClient = Depends(get_db)):
        self._db: PostgreClient = db_client

    def close(self):
        self._db.close()