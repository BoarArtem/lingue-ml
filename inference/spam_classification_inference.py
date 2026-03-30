import torch
from models.spam_classification_model import SpamClassificationModel
from data.spam_classification.dataset import vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SpamClassificationModel(vocab_size=10000, embed_dim=128, hidden_size=256, num_layers=2).to(device)
model.load_state_dict(torch.load('spam_classification_model.pth'))
model.eval()

def encode_text(text, vocab, max_len):
    tokens = text.lower().split()
    encoded = [vocab.get(w, 0) for w in tokens]

    encoded = encoded[:max_len]

    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))

    return torch.tensor(encoded).unsqueeze(0)

text = "sdfasfdfa drug killers"

x = encode_text(text, vocab, max_len=128)
x = x.long().to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)

print("Class:", pred.item())
print("Probabilities:", probs.tolist())
