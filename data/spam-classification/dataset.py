import pickle
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from data.preprocess import spam_classification_preprocess

data = spam_classification_preprocess("../datasets/spam_Emails_data.csv")

X = data['text']
y = data['labels']


with open("../tokens_cache.pkl", "rb") as f:
    tokens = pickle.load(f)

all_tokens = [w for text in tokens for w in text]

counter = Counter(all_tokens)

vocab = {
    w: i+1
    for i, (w, _) in enumerate(counter.most_common(10000))
}


# encoded text
encoded_text = [
    torch.tensor([vocab.get(w, 0) for w in text])
    for text in tokens
]

# pad sequence
padded = pad_sequence(encoded_text, batch_first=True, padding_value=0)

# embedding
embedding = torch.nn.Embedding(
    num_embeddings=len(vocab) + 1,
    embedding_dim=128,
    padding_idx=0
)

class SpamDataset(Dataset):
    def __init__(self, vocab, tokens, labels, max_len=None):
        self.vocab = vocab
        self.labels = labels
        self.max_len = max_len

        self.encoded = [
            torch.tensor([vocab.get(w, 0) for w in text])
            for text in tokens
        ]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = self.encoded[idx]

        if self.max_len:
            seq = seq[:self.max_len]

        label = torch.tensor(self.labels[idx])

        return seq, label

def collate_fn(batch):
    sequence, labels = zip(*batch)

    padded_sequences = pad_sequence(
        sequence,
        batch_first=True,
        padding_value=0
    )

    labels = torch.tensor(labels)

    return padded_sequences, labels


dataset = SpamDataset(vocab, tokens, y, max_len=128)
