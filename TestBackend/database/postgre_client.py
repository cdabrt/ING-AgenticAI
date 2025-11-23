import logging
from typing import List

from contracts.models import RequirementBundle

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

    def get_requirement_bundles(self, bundle_ids: List[int]) -> List[RequirementBundle]:
        """
        Retrieve RequirementBundle records by their document IDs.
        Args:
            bundle_ids (List[int]): List of document IDs to filter RequirementBundles.
        Returns:
            List[RequirementBundle]: List of RequirementBundle records.
        """
        if(len(bundle_ids) == 0):
            return self.db.query(RequirementBundle).all()
        return self.db.query(RequirementBundle).filter(RequirementBundle.document.in_(bundle_ids)).all()
    
    def get_requirement_bundle_by_id(self, bundle_id: int) -> RequirementBundle:
        """
        Retrieve a single RequirementBundle record by its document ID.
        Args:
            bundle_id (int): Document ID of the RequirementBundle to retrieve.
        Returns:
            RequirementBundle: The RequirementBundle record.
        """
        return self.db.query(RequirementBundle).filter(RequirementBundle.document == bundle_id).first()

    def save_requirement_bundle(self, bundle: RequirementBundle) -> RequirementBundle:
        """
        Save a RequirementBundle record to the database.
        Args:
            bundle (RequirementBundle): The RequirementBundle record to save.
        Returns:
            RequirementBundle: The saved RequirementBundle record.
        """
        self.db.add(bundle)
        self.db.commit()
        self.db.refresh(bundle)
        return bundle
    
    def delete_requirement_bundle(self, bundle: RequirementBundle):
        """
        Delete a RequirementBundle record from the database.
        Args:
            bundle (RequirementBundle): The RequirementBundle record to delete.
        """
        self.db.delete(bundle)
        self.db.commit()

    def flush(self):
        self.db.flush()

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()