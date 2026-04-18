# models/disease_model.py
# Plant disease detection — auto-detects any .keras model in models/ folder
# Works with any filename: "trained_model (1).keras", "disease.keras", etc.

import os, json, numpy as np

_model = None
_class_names = []

MODELS_DIR       = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")


def _find_keras_model():
    """Auto-find any .keras file in models/ — no matter what it is named."""
    preferred = ["trained_model.keras", "plant_disease_model.keras", "disease_model.keras"]
    for name in preferred:
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            return p
    # Pick the first .keras file found (handles "trained_model (1).keras" etc.)
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".keras"):
            return os.path.join(MODELS_DIR, f)
    return None


def _find_class_names():
    """Auto-find class names JSON in models/."""
    if os.path.exists(CLASS_NAMES_PATH):
        return CLASS_NAMES_PATH
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".json") and "class" in f.lower():
            return os.path.join(MODELS_DIR, f)
    # Last resort: any json file
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".json"):
            return os.path.join(MODELS_DIR, f)
    return None


def load_disease_model():
    global _model, _class_names
    # --- Load .keras model ---
    try:
        from tensorflow.keras.models import load_model
        model_path = _find_keras_model()
        if model_path:
            _model = load_model(model_path)
            print(f"[DiseaseModel] Loaded: {os.path.basename(model_path)}")
        else:
            print(f"[DiseaseModel] WARNING: No .keras file found in models/")
            print(f"[DiseaseModel] Place your model file inside the models/ folder.")
    except Exception as e:
        print(f"[DiseaseModel] Error loading model: {e}")

    # --- Load class names ---
    try:
        json_path = _find_class_names()
        if json_path:
            with open(json_path, "r") as f:
                _class_names = json.load(f)
            print(f"[DiseaseModel] {len(_class_names)} classes loaded from {os.path.basename(json_path)}")
        else:
            print(f"[DiseaseModel] WARNING: No class_names.json found in models/")
    except Exception as e:
        print(f"[DiseaseModel] Error loading class names: {e}")


def predict_disease(img_path: str):
    """Returns (label, confidence_pct, top5_list)"""
    if _model is None or not _class_names:
        return "Model not loaded", 0.0, []
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=(128, 128), color_mode="rgb")
    arr = image.img_to_array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    preds = _model.predict(arr, verbose=0)[0]
    top5_idx = preds.argsort()[-5:][::-1]
    top5 = [(_class_names[i], round(float(preds[i]) * 100, 2)) for i in top5_idx]
    return top5[0][0], top5[0][1], top5
