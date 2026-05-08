'''Лемматизация и токенизация'''
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def b2_time_prediction_preprocess(filepath):
    df = pd.read_csv(filepath)
    target_col = 'target_days_b2'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    
    clean_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]
    return " ".join(clean_tokens)


def preprocess_text(text):
    """
    :param text: Поле с текстами в датасете
    :return: Возвращает процесс очистки текста
    """
    text = str(text)

    # main text label preprocess
    text = re.sub(r'escapenumber|escapelong', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def spam_classification_preprocess(filepath):
    """
    :param filepath: Путь до файла датасета
    :return: Возвращает колонки и внутренности датасета
    """
    data = pd.read_csv(filepath)

    data['label'] = data['label'].map({'Spam': 1, 'Ham': 0}) # encoding label data like 0 and 1, where 1 - spam, 0 - ham
    data['text'] = data['text'].astype(str).apply(preprocess_text)

    return data


if __name__ == "__main__":
    data = spam_classification_preprocess("datasets/spam_Emails_data.csv")

    print(data.head())
