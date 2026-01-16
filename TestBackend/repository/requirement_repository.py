from database.database_client import DatabaseClient
from AgenticAI.agentic.models import RequirementItem as RequirementItemLocalModel
from database.models.models import RequirementItem as RequirementItemDatabaseModel, RequirementBundle as RequirementBundleDatabaseModel
from database.models.adapters.requirement_item_adapter import adapt_to_model, adapt_to_db_model

class RequirementRepository(DatabaseClient):

    def get_requirement_item(self, requirement_id: str) -> RequirementItemLocalModel:
        model_item = self._get_db_requirement_item(requirement_id)
        return adapt_to_model(model_item)
    
    def save_requirement_item(self, requirement: RequirementItemLocalModel) -> RequirementItemLocalModel:
        db_requirement = self._get_db_requirement_item(requirement.id)

        if db_requirement is None:
            return None

        db_requirement.description = requirement.description
        db_requirement.rationale = requirement.rationale

        saved_item = self._db.save(db_requirement)
        return adapt_to_model(saved_item)
        
    def _get_db_requirement_item(self, requirement_id: str) -> RequirementItemDatabaseModel:
        return self._db.get_first(RequirementItemDatabaseModel, req_id=requirement_id)
