# models/crop_model.py
# Crop recommendation using best_crop_model.pkl (RandomForest)
# Features: Temperature, Humidity, pH, Rainfall
# Also needs: crop_scaler.pkl and crop_label_encoder.pkl

import os, numpy as np

_model = None
_scaler = None
_label_encoder = None

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(MODELS_DIR, "best_crop_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "crop_scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "crop_label_encoder.pkl")


def load_crop_model():
    global _model, _scaler, _label_encoder
    import joblib
    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            print("[CropModel] loaded best_crop_model.pkl")
        else:
            print(f"[CropModel] WARNING: {MODEL_PATH} not found.")

        if os.path.exists(SCALER_PATH):
            _scaler = joblib.load(SCALER_PATH)
            print("[CropModel] loaded crop_scaler.pkl")
        else:
            print(f"[CropModel] WARNING: {SCALER_PATH} not found.")

        if os.path.exists(ENCODER_PATH):
            _label_encoder = joblib.load(ENCODER_PATH)
            print("[CropModel] loaded crop_label_encoder.pkl")
        else:
            print(f"[CropModel] WARNING: {ENCODER_PATH} not found.")

    except Exception as e:
        print(f"[CropModel] Error loading: {e}")


def predict_crop(temperature: float, humidity: float, ph: float, rainfall: float):
    """
    Predict best crop using ML model.
    Returns (crop_name: str, top5: list of (crop, probability_pct))
    """
    if _model is None:
        return None, []

    sample = np.array([[temperature, humidity, ph, rainfall]])

    if _scaler is not None:
        sample = _scaler.transform(sample)

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(sample)[0]
        top5_idx = proba.argsort()[-5:][::-1]
        if _label_encoder is not None:
            top5 = [(_label_encoder.inverse_transform([i])[0], round(float(proba[i]) * 100, 2)) for i in top5_idx]
        else:
            top5 = [(str(i), round(float(proba[i]) * 100, 2)) for i in top5_idx]
        return top5[0][0], top5
    else:
        pred_encoded = _model.predict(sample)
        if _label_encoder is not None:
            crop_name = _label_encoder.inverse_transform(pred_encoded)[0]
        else:
            crop_name = str(pred_encoded[0])
        return crop_name, [(crop_name, 100.0)]


def model_loaded() -> bool:
    return _model is not None
