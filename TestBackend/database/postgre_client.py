import logging

try:
     from database.session import SessionLocal
except:
    SessionLocal = lambda: logging.error("Session could not be imported!")

def get_db():
    db = PostgreClient()
    try:
        yield db
    finally:
        db.close()

class PostgreClient:
    def __init__(self):
        self.db = SessionLocal()
    
    def delete(self, item):
        self.db.delete(item)
        self.db.commit()

    def save(self, item):
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def get_first(self, model, **filters):
        return self.db.query(model).filter_by(**filters).first()
    
    def get_all(self, model, **filters):
        query = self.db.query(model)
        if filters:
            query = query.filter_by(**filters)
        return query.all()
    
    def get_all_deferred(self, model, *defer_options, **filters):
        """Get all items with deferred loading for specific columns"""
        query = self.db.query(model).options(*defer_options)
        if filters:
            query = query.filter_by(**filters)
        return query.all()

    def flush(self):
        self.db.flush()

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()