import joblib

class TopicPredictor:
    def __init__(self):
        self.model = joblib.load("lingue-ml/inference/topic_model.pkl")
        self.vectorizer = joblib.load("lingue-ml/inference/topic_vectorizer.pkl")

    def predict(self, text: str):
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)

        best_idx = probs[0].argmax()
        best_score = probs[0][best_idx]
        best_label = self.model.classes_[best_idx]

        if best_score > 0.4:
            return best_label
        
        return None