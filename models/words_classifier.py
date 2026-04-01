from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.datasets.topic_dataset import generate_dataset


def train():
    texts, labels = generate_dataset(50)
    print("Training topic model...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=200)
    model.fit(X, labels)

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    joblib.dump(model, os.path.join(BASE_DIR, "inference", "topic_model.pkl"))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, "inference", "topic_vectorizer.pkl"))

    print("Topic model trained!")
    
def predict(text):
    
    model = joblib.load("lingue-ml/inference/topic_model.pkl")
    vectorizer = joblib.load("lingue-ml/inference/topic_vectorizer.pkl")
    
    
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    
    return prediction[0]

if __name__ == "__main__":
    train()