import os
import joblib
import pandas as pd
from models.b2_predictor import B2PredictorModel

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "inference",
    "b2_model.pkl"
)


def test_b2_model_load():
    assert os.path.exists(MODEL_PATH), f"Модель не найдена: {MODEL_PATH}"

    predictor: B2PredictorModel = joblib.load(MODEL_PATH)

    assert predictor is not None
    assert isinstance(predictor, B2PredictorModel)

    assert predictor.model is not None
    assert isinstance(predictor.feature_names, list)
    assert len(predictor.feature_names) > 0


def test_b2_model_inference():
    predictor: B2PredictorModel = joblib.load(MODEL_PATH)

    df = pd.DataFrame(
        [[0] * len(predictor.feature_names)],
        columns=predictor.feature_names
    )

    prediction = predictor.predict(df)[0]

    assert prediction is not None
    assert isinstance(prediction, (float, int))