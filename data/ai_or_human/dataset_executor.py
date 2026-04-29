import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def get_csv_x_y(filepath):
    df = pd.read_csv(filepath, index_col=0)
    X = df["text"]
    y = df["human_or_ai"]

    return X, y

def lemmatize_text(text):

    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]

    return " ".join(lemmatized_text)

def tokenize_text(text):
    return word_tokenize(text)

class AIOrHumanDataset(Dataset):
    def __init__(self, filepath):
        self.X, self.y = get_csv_x_y(filepath)
        self.vocab = 1000

    def __getitem__(self, idx):
        lemmatized_text = lemmatize_text(self.X[idx])
        tokenized_text = tokenize_text(lemmatized_text)

        return tokenized_text, self.y[idx]

    def __len__(self):
        return len(self.X)

def get_dataloader(filepath, batch_size=32, shuffle=True):
    dataset = AIOrHumanDataset(filepath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # print(get_dataloader("ai_human_detection_v1.csv"))