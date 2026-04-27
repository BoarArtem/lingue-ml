import pandas as pd

def get_csv_x_y(filepath):
    df = pd.read_csv(filepath, index_col=0)
    X = df["text"]
    y = df["human_or_ai"]

    return X, y


