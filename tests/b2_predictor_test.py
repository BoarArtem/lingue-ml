import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "inference", "b2_model.pkl")

def test_b2_model_load():
    assert os.path.exists(MODEL_PATH), f"Модель не найдена: {MODEL_PATH}"

    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)

    assert model is not None

def test_b2_model_inference():
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)

    prediction = model.predict([[0]*12])
    assert prediction is not None