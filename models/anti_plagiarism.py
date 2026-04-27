import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim * 2, 1) # *2 for bidirectional

    def forward(self, outputs):
        # outputs: [B, T, 2H]
        scores = self.score(outputs) # [B, T, 1]
        weights = torch.softmax(scores, dim=1) # [B, T, 1]

        context = (weights * outputs).sum(dim=1) # [B, 2H]
        return context, weights

class AntiPlagiarismModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: [B, T]
        emb = self.embedding(x) # [B, T, embed_dim]
        outputs, _ = self.encoder(emb)

        context, attn_weights = self.attention(outputs)

        out = self.fc(context) # [B, output_dim]

        return out, attn_weights

def get_anti_plagiarism_model(vocab_size, embed_dim, hidden_dim, output_dim):
    return AntiPlagiarismModel(vocab_size, embed_dim, hidden_dim, output_dim)

def load_anti_plagiarism_model(path, vocab_size, embed_dim, hidden_dim, output_dim):
    model = get_anti_plagiarism_model(vocab_size, embed_dim, hidden_dim, output_dim)

    model.load_state_dict(torch.load(path, map_location=DEVICE))

    return model

def train_anti_plagiarism_model(num_epochs, model, optimizer, loss_fn, train_loader):
    model.to(DEVICE).train()

    for epoch in range(num_epochs):

        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            out, attn_weights = model(x)

            loss = loss_fn(out, y)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"anti_plagiarism_model_epoch_{epoch}.pth")
            print(f"Saved model : [anti_plagiarism_model_epoch_{epoch}.pth] at epoch {epoch}")