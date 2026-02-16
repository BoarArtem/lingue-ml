import joblib
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocess import b2_time_prediction_preprocess
from models.b2_predictor import B2PredictorModel

DATA_PATH = 'data/dataset_b2.csv'
MODEL_PATH = 'inference/b2_model.pkl'

def train_and_save():

    X_train, X_test, y_train, y_test = b2_time_prediction_preprocess(DATA_PATH)
    predictor = B2PredictorModel()
    predictor.train(X_train, y_train)
    predictor.evaluate(X_test, y_test)
    predictor.show_feature_importance()
    
    joblib.dump(predictor.get_model(), MODEL_PATH)


def predict_days(user_data):
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    

    model = joblib.load(MODEL_PATH)
    

    df = pd.DataFrame([user_data])
    days = model.predict(df)[0]
    
    return int(days)


if __name__ == "__main__":

    if not os.path.exists(DATA_PATH):
        print(f"Нет файла {DATA_PATH}")
    else:
        
        test_user = {
            'unique_words': 1000,
            'words_a1': 50, 'words_a2': 5, 'words_b1': 5, 'words_b2': 0,
            'avg_acc_7d': 0.7, 'avg_acc_30d': 0.8, 
            'avg_time_sec': 4.0,
            'words_day_7d': 14, 'words_day_30d': 140,
            'streak': 4, 'sessions_week': 9
        }
        
        print("\nТестовый прогноз юзера:")
        result = predict_days(test_user)
        print(f"До уровня B2 осталось примерно: {result} дней")