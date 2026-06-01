import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import torch

def retrieve_audio_text_pairs(main_directory: str, key_name: str) -> list[tuple[str, str]]:
    """Function that retrieves audio and text pairs from a directory."""

    audio_dir = os.path.join(main_directory, key_name)
    tsv_path = os.path.join(main_directory, f"{key_name}.tsv")

    if not os.path.isdir(audio_dir) or not os.path.isfile(tsv_path):
        return []

    tsv_df = get_tsv_content(tsv_path)
    # TSV columns: id, filename, transcription, lowercase_transcription, phonemes, num_samples, gender
    filename_to_transcript = {row.iloc[1]: row.iloc[2] for _, row in tsv_df.iterrows()}

    pairs = []
    for file in sorted(os.listdir(audio_dir)):
        if file.endswith(".wav") and file in filename_to_transcript:
            audio_path = os.path.join(audio_dir, file)
            pairs.append((audio_path, filename_to_transcript[file]))

    return pairs

def get_tsv_content(filepath):
    """Function that retrieves content from a TSV file."""
    return pd.read_csv(filepath, sep='\t', header=None)

def get_mel_spectogram(audio_path: str) -> torch.Tensor:
    """Function that retrieves mel spectrogram from audio file"""
    wav, sr = librosa.load(audio_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )

    mel = np.log(np.clip(mel, 1e-5, None))

    return torch.from_numpy(mel).float()

def text_to_sequence(text: str) -> list[int]:
    """Converts text to a sequence of character indices."""
    chars = "abcdefghijklmnopqrstuvwxyz '-.,!?;:"
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    return [char_to_idx.get(c, 0) for c in text.lower()]

class FleursDataset(Dataset):
    """Dataset class for Fleurs dataset."""
    def __init__(self, dir_name: str, key_name: str):
        self.dir_name = dir_name
        self.key_name = key_name
        self.pairs = retrieve_audio_text_pairs(dir_name, key_name)

        self.mel_spectograms = [get_mel_spectogram(audio_path) for audio_path, _ in self.pairs]
        self.sequences = [text_to_sequence(text) for _, text in self.pairs]

    def __len__(self):
        return len(self.mel_spectograms)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        mel_spectogram = self.mel_spectograms[idx]

        return sequence, mel_spectogram

def collate_fn(batch):
    """Pads variable-length text sequences and mel spectrograms to the max length in the batch."""
    sequences, mels = zip(*batch)

    # Pad text sequences to max length in batch
    max_text_len = max(s.size(0) for s in sequences)
    padded_sequences = torch.zeros(len(sequences), max_text_len, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded_sequences[i, :s.size(0)] = s

    # Pad mel spectrograms to max time length in batch (shape: [n_mels, T])
    max_mel_len = max(m.size(1) for m in mels)
    padded_mels = torch.zeros(len(mels), mels[0].size(0), max_mel_len)
    for i, m in enumerate(mels):
        padded_mels[i, :, :m.size(1)] = m

    return padded_sequences, padded_mels

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Function that retrieves a DataLoader from a Dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def extract_dict_of_loaders(dir_name: str, train_key: str, test_key: str, batch_size: int) -> dict[str, DataLoader]:
    """Function that extracts a dictionary of DataLoaders from a list of dataset directories."""
    pair_dict = {}

    for dataset_dir in os.listdir(dir_name):
        dataset_dir_path = os.path.join(dir_name, dataset_dir)
        if not os.path.isdir(dataset_dir_path):
            continue

        train_dataset = FleursDataset(dataset_dir_path, train_key)
        test_dataset = FleursDataset(dataset_dir_path, test_key)

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"Skipping '{dataset_dir}': train={len(train_dataset)}, test={len(test_dataset)} samples")
            continue

        pair_dict[dataset_dir] = {
            "train": get_dataloader(train_dataset, batch_size, shuffle=True),
            "test": get_dataloader(test_dataset, batch_size, shuffle=False)
        }

    return pair_dict

if __name__ == "__main__":
    dir_name = "../fleurs"

    train_key = "train"
    test_key = "test"
    batch_size = 32

    pairs = extract_dict_of_loaders(dir_name, train_key, test_key, batch_size)

    print(pairs)