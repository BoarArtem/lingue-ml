import os
import torch
from models.spam_classification_model import SpamClassificationModel
from data.spam_classification.dataset import vocab

model_dir = os.getenv("MODEL_DIR", "/models")  # for docker testing/production

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpamClassificationModel(
    vocab_size=10000,
    embed_dim=128,
    hidden_size=256,
    num_layers=2,
).to(device)
model.load_state_dict(torch.load(f"{model_dir}/spam_classification_model.pth", map_location=device))
model.eval()

def spam_or_ham(sentence):
    def encode_text(text, vocab, max_len):
        tokens = text.lower().split()

        encoded = [vocab.get(w, 0) for w in tokens]
        encoded = encoded[:max_len]

        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))

        return torch.tensor(encoded).unsqueeze(0)

    # final sentence settings
    x = encode_text(sentence, vocab, max_len=128)
    x = x.long().to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=-1)

    label_map = {1: "spam", 0: "ham"}

    return f"Class: {label_map[pred.item()]}"

if __name__ == "__main__":
    print(spam_or_ham("My family very love cars"))

