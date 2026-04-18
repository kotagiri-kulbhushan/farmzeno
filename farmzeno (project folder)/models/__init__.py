# models/__init__.py
from models.disease_model import load_disease_model, predict_disease
from models.crop_model import load_crop_model, predict_crop, model_loaded

__all__ = [
    "load_disease_model", "predict_disease",
    "load_crop_model", "predict_crop", "model_loaded",
]
