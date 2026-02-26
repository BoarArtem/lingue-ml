import joblib
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


MODEL_PATH = 'lingue-ml/inference/b2_model.pkl'

def predict_days(user_data):
    if not os.path.exists(MODEL_PATH):
        return "Ошибка: Файл модели .pkl не найден! Сначала нужно запустить обучение."
    
    model = joblib.load(MODEL_PATH)
    
    df = pd.DataFrame([user_data])
    
    days = model.predict(df)[0]
    return int(days)

if __name__ == "__main__":
   
    test_user = {
        'unique_words': 1500, 
        'words_a1': 600, 'words_a2': 500, 'words_b1': 400, 'words_b2': 0,
        'avg_acc_7d': 0.88, 'avg_acc_30d': 0.85, 
        'avg_time_sec': 6.0,
        'words_day_7d': 30, 'words_day_30d': 900,
        'streak': 20, 'sessions_week': 14
    }
    
    result = predict_days(test_user)
    print(f"До уровня B2 осталось примерно: {result} дней")