import os
import sys
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.topic_classifier import train_topic_model
from data.preprocess import clean_text

class TopicPredictor:
    def __init__(self):
        self.model_path = os.path.join(current_dir, "topic_model.pkl")
        self.dataset_path = os.path.join(root_dir, "data", "datasets", "topic_dataset.csv")
        self.model = self._load_or_train()


    def _load_or_train(self):
        if not os.path.exists(self.model_path):
            print("Модель не найдена!")
            if not os.path.exists(self.dataset_path):
                print("Датасет не найден")
                sys.exit(1)
            train_topic_model()
        return joblib.load(self.model_path)

    def predict(self, text: str):
        """Возвращает предсказанную тему и процент уверенности(для тестов)."""
        clean_str = clean_text(text)
        prediction = self.model.predict([clean_str])[0]
        probabilities = self.model.predict_proba([clean_str])[0]
        confidence = max(probabilities) * 100
        return prediction, confidence

    def get_topic(self, text: str) -> str:
        """Возвращает только название темы (без процентов)."""
        prediction, _ = self.predict(text)
        return prediction

    def get_topics(self, sentences: list) -> list:
        """Принимает список фраз, возвращает список тем для каждой (для API)."""
        cleaned_sentences = [clean_text(text) for text in sentences]
        predictions = self.model.predict(cleaned_sentences)
        return predictions.tolist()