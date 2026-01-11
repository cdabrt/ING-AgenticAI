from typing import List, Optional
from database.database_provider import DatabaseProvider
from AgenticAI.agentic.models import RequirementBundle
from contracts.adapters.requirement_bundle_adapter import adapt_to_model, adapt_to_contract

class BundleProvider(DatabaseProvider):

    def get_requirement_bundle(self, bundle_id: str) -> RequirementBundle:
        """Get a single requirement bundle by ID"""
        db_bundle = self._db.get_requirement_bundle_by_id(bundle_id)
        return adapt_to_model(db_bundle) if db_bundle else None
    
    def get_requirement_bundles(self, bundle_ids: Optional[List[int]] = None) -> List[RequirementBundle]:
        """Get all requirement bundles or filter by IDs"""
        if bundle_ids is None:
            return [adapt_to_model(b) for b in self._db.get_requirement_bundles([])]
        return [adapt_to_model(b) for b in self._db.get_requirement_bundles(bundle_ids)]
    
    def save_requirement_bundle(self, bundle: RequirementBundle) -> RequirementBundle:
        contract_bundle = adapt_to_contract(bundle)
        saved_bundle = self._db.save_requirement_bundle(contract_bundle)
        return adapt_to_model(saved_bundle)