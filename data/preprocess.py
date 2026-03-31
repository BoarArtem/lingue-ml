'''Летимизация и токенизация'''
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split


nltk.download('punkt')
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

