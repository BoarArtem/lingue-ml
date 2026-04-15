'''Скрипт обучения ML-модели'''
import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)


from data.preprocess import clean_text

def train_topic_model():
    
    dataset_path = os.path.join(root_dir, "data", "datasets", "topic_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Ошибка: Датасет не найден ({dataset_path})")
        return

    df = pd.read_csv(dataset_path)
    

    df['clean_text'] = df['text'].apply(clean_text)
    
    X = df['clean_text']
    y = df['label']
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, max_iter=2000))
    ])
    
    pipeline.fit(X_train, y_train)
    

    y_pred = pipeline.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred))
    

    inference_dir = os.path.join(root_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    model_path = os.path.join(inference_dir, "topic_model.pkl")
    joblib.dump(pipeline, model_path)
    
    print(f"Готово: {model_path}")

if __name__ == "__main__":
    train_topic_model()