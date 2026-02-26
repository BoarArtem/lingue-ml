import os
import joblib
import pandas as pd
from models.b2_predictor import B2PredictorModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "inference", "b2_model.pkl")

def test_b2_model_load():
    assert os.path.exists(MODEL_PATH), f"Модель не найдена: {MODEL_PATH}"

    with open(MODEL_PATH, "rb") as f:
        predictor: B2PredictorModel = joblib.load(f)

    assert predictor is not None
    assert isinstance(predictor, B2PredictorModel)
    assert hasattr(predictor, "model")
    assert hasattr(predictor, "feature_names")

def test_b2_model_inference():
    with open(MODEL_PATH, "rb") as f:
        predictor: B2PredictorModel = joblib.load(f)

    df = pd.DataFrame([[0]*len(predictor.feature_names)], columns=predictor.feature_names)

    prediction = predictor.model.predict(df)[0]

    assert prediction is not None
    assert isinstance(prediction, float) or isinstance(prediction, int)