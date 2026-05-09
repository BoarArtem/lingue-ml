from earlystopping import EarlyStopping
import torch

callbacks = [EarlyStopping(patience=5)]
model: torch.nn.Module

def apply_callbacks(train_loader, total_loss):
    for callback in callbacks:
        callback(total_loss / len(train_loader))

        if callback.early_stop:
            print("Early stopping...")
            return