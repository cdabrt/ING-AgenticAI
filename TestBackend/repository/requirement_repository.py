from TestBackend.database.database_client import DatabaseClient
from AgenticAI.agentic.models import RequirementItem as RequirementItemLocalModel
from database.models.models import RequirementItem as RequirementItemDatabaseModel
from database.models.adapters.requirement_item_adapter import adapt_to_model, adapt_to_db_model

class RequirementRepository(DatabaseClient):

    def get_requirement_item(self, requirement_id: str) -> RequirementItemLocalModel:
        model_item = self._db.get_first(RequirementItemDatabaseModel, id=requirement_id)
        return adapt_to_model(model_item)
    
    def save_requirement_item(self, requirement: RequirementItemLocalModel) -> RequirementItemLocalModel:
        model_item = adapt_to_db_model(requirement)
        saved_item = self._db.save(model_item)
        return adapt_to_model(saved_item)