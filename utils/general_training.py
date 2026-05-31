import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


callbacks = [EarlyStopping(patience=5)]
model: torch.nn.Module

def apply_callbacks(train_loader, total_loss):
    for callback in callbacks:
        callback(total_loss / len(train_loader))

        if callback.early_stop:
            print("Early stopping...")
            return