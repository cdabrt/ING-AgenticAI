from typing import List, Optional
from database.database_client import DatabaseClient
from database.models.models import RequirementBundle as RequirementBundleDatabaseModel 
from AgenticAI.agentic.models import RequirementBundle as RequirementBundleLocalModel
from database.models.adapters.requirement_bundle_adapter import adapt_to_db_model, adapt_to_model

class BundleRepository(DatabaseClient):
    def get_requirement_bundle(self, bundle_id: str) -> RequirementBundleLocalModel:
        """Get a single requirement bundle by ID"""
        return self._db.get_first(RequirementBundleDatabaseModel, id=bundle_id)
    
    def get_requirement_bundles(self, bundle_ids: Optional[List[int]] = None) -> List[RequirementBundleLocalModel]:
        """Get all requirement bundles or filter by IDs"""
        if bundle_ids is None:
            return [adapt_to_model(bundle) for bundle in self._db.get_all(RequirementBundleDatabaseModel)]
        return [adapt_to_model(bundle) for bundle in self._db.get_all(RequirementBundleDatabaseModel, id=bundle_ids)]
    
    def save_requirement_bundle(self, bundle: RequirementBundleLocalModel) -> RequirementBundleLocalModel:
        model_bundle = adapt_to_db_model(bundle)
        saved_bundle = self._db.save(model_bundle)
        return adapt_to_model(saved_bundle)