import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from models.b2_predictor import B2PredictorModel


MODEL_PATH = "b2_model.pkl"


def train_model(data_path: str):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["target_days_b2"])
    y = df["target_days_b2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    predictor = B2PredictorModel()
    predictor.train(X_train, y_train)
    predictor.evaluate(X_test, y_test)

    os.makedirs("inference", exist_ok=True)
    joblib.dump(predictor, MODEL_PATH)

    print(f"Модель сохранена в {MODEL_PATH}")


def predict_days(user_data: dict) -> int:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Файл модели не найден. Сначала нужно обучить модель."
        )

    predictor: B2PredictorModel = joblib.load(MODEL_PATH)

    df = pd.DataFrame([user_data])
    days = predictor.predict(df)[0]

    return int(days)

if __name__ == "__main__":

    train_model("../data/datasets/dataset_b2.csv")

    test_user = {
        'unique_words': 1500,
        'words_a1': 600,
        'words_a2': 500,
        'words_b1': 400,
        'words_b2': 0,
        'avg_acc_7d': 0.88,
        'avg_acc_30d': 0.85,
        'avg_time_sec': 6.0,
        'words_day_7d': 30,
        'words_day_30d': 900,
        'streak': 20,
        'sessions_week': 14
    }

    result = predict_days(test_user)
    print(f"До уровня B2 осталось примерно: {result} дней")