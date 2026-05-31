import pickle
from collections import Counter
import torch.nn
from torch.nn.functional import embedding
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from data.preprocess import spam_classification_preprocess

data = spam_classification_preprocess("../data/datasets/spam_Emails_data.csv")

X = data['text']
y = data['label']

with open("../data/tokens_cache.pkl", "rb") as f:
    tokens = pickle.load(f)

all_tokens = [w for text in tokens for w in text]

# counter of token items
counter = Counter(all_tokens)

# vocab of tokens
vocab = {
    w: i+1
    for i, (w, _) in enumerate(counter.most_common(10000))
}


# main dataset class
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

        labels = torch.tensor(self.labels[idx])

        return seq, labels

def collate_fn(batch):
    sequence, labels = zip(*batch)

    padded_sequence = pad_sequence(
        sequence,
        batch_first=True,
        padding_value=0
    )

    labels = torch.tensor(labels)

    return padded_sequence, labels


# creating dataset object
dataset = SpamDataset(vocab, tokens, y, 128)


# creating dataloader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# main testing (1)
# for X, y in loader:
#     print(X.shape)
#     print(y.shape)
#     break

# (2)
# print(X[0][:10])
# print(y[0])

# embedding testing
# embedding = torch.nn.Embedding(
#     num_embeddings=len(vocab) + 1,
#     embedding_dim=128,
#     padding_idx=0
# )
#
# emb = embedding(X)
# print(emb)