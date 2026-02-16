import pandas as pd
from sklearn.model_selection import train_test_split

def vocabulary_expander_preprocess():
    with open("./datasets/words-en.txt", "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
        words = [w for w in words if len(w) >= 3]

        words =  list(set(words))

    return words

def b2_time_prediction_preprocess(filepath):
    df = pd.read_csv(filepath)
    target_col = 'target_days_b2'
    
    X = df.drop(columns=[target_col]) 
    y = df[target_col]               
    
    return train_test_split(X, y, test_size=0.2, random_state=42)