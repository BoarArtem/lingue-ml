import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
import torch

lemmatizer = WordNetLemmatizer()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_csv_x_y(filepath, x_label="text", y_label="human_or_ai"):
    df = pd.read_csv(filepath, index_col=0)
    X = df[x_label]
    y = df[y_label]

    return X, y

def lemmatize_text(text):

    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]

    return " ".join(lemmatized_text)

def tokenize_text(text):
    return tokenizer.tokenize(text)

def encode_text(text):
    return tokenizer.encode(text, add_special_tokens=True)

def decode_text(text):
    return tokenizer.decode(text)

def encode_user_text(user_text):
    encoded = encode_text(user_text)
    return torch.tensor(encoded)

class AIOrHumanDataset(Dataset):
    def __init__(self, filepath, x_label="text", y_label="human_or_ai"):
        self.X, self.y = get_csv_x_y(filepath, x_label=x_label, y_label=y_label)
        self.vocab = tokenizer.vocab_size

    def __getitem__(self, idx):
        encoded_text = encode_user_text(self.X.iloc[idx])
        return encoded_text, self.y.iloc[idx]

    def __len__(self):
        return len(self.X)

LABEL_MAP = {"human": 0, "ai": 1, "post_edited_ai": 2}

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id or 0)
    numeric_labels = [LABEL_MAP[label] for label in labels]
    return padded, torch.tensor(numeric_labels)

def get_dataloader(filepath, batch_size=32, shuffle=True):
    dataset = AIOrHumanDataset(filepath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)