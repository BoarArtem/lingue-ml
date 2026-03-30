import torch.optim

from data.spam_classification import dataset
from torch import nn
from data.spam_classification.dataset import loader

# model class
class SpamClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.3):
        super(SpamClassificationModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # embedding layer
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        # main lstm
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # output layer
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden_last = hidden[-1]
        out = self.dropout(hidden_last)
        logits = self.fc(out)

        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SpamClassificationModel(vocab_size=10000, embed_dim=128, hidden_size=256, num_layers=2).to(device)

# loss
criterion = nn.CrossEntropyLoss()

# optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# num of epochs
epochs = 5

model.train()

if __name__ == "__main__":
    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device).long()
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), '../inference/spam_classification_model.pth')
