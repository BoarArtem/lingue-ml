import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def b2_time_prediction_preprocess(filepath):
    df = pd.read_csv(filepath)
    target_col = 'target_days_b2'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def spam_classification_preprocess(filepath):
    data = pd.read_csv(filepath)

    data['label'] = data['label'].map({'Spam': 1, 'Ham': 0}) # encoding label data like 0 and 1, where 1 - spam, 0 - ham

    return data
