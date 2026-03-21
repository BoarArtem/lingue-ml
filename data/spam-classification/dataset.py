import pickle
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

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

