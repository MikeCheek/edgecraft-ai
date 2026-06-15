# app/services/shared_state.py
from app.services.data_manager import DataManager
from app.services.trainer import Trainer
from app.services.converter import Converter

# These instances will persist for the lifetime of the FastAPI app
data_manager = DataManager()
trainer = Trainer()
converter = Converter()