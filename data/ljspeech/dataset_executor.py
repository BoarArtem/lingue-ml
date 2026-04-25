import pandas as pd
from pymorphy3.analyzer import auto
from torch.utils.data import Dataset, DataLoader

import torch

import librosa
import numpy as np
import os

import logging

def apply_df_with_file_verification(csv_df: str, wav_full_path: str):
    """Function that checks if the wav files match with the lines of metadata.csv"""
    df = execute_csv(f"{csv_df}")
    existing_files = sorted(os.listdir(f"{wav_full_path}"))
    existing_stems = {f.split(".")[0] for f in existing_files}
    executable_files = []
    rows = []

    for existing_file_name in existing_files:
        stem = existing_file_name.split(".")[0]
        match = df[df[0] == stem]
        if not match.empty:
            executable_files.append(existing_file_name)
            rows.append(match)

    executable_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)

    return executable_df, executable_files


def execute_csv(csv_path) -> pd.DataFrame:
    """Function that loads columns from csv file"""
    try:
        df = pd.read_csv(csv_path, sep="|", header=None, names=None, quoting=3)
        df = df.dropna(subset=[2]).reset_index(drop=True)
        return df
    except Exception as e:
        logging.info(f"Error loading CSV file: {e}")
        raise

def get_col_data(df, col_name) -> list:
    """Function that retrieves column data as a list"""
    return df[col_name].tolist()

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

class LJSpeechDataset(Dataset):
    def __init__(self, dataset_path: str, audio_dir_path: str):
        df, _ = apply_df_with_file_verification(dataset_path, audio_dir_path)

        # if len(os.listdir(audio_dir_path)) != len(df):
        #     raise Exception("Number of audio files and metadata files are not equal")

        self.input_col_data = [text_to_sequence(t) for t in get_col_data(df, 2)]
        self.target_col_data = [get_mel_spectogram(os.path.join(audio_dir_path, audio_path)) for audio_path in os.listdir(audio_dir_path)]

    def __getitem__(self, idx):
        return torch.tensor(self.input_col_data[idx], dtype=torch.long), self.target_col_data[idx]

    def __len__(self):
        return len(self.input_col_data)

def collate_fn(batch):
    texts, mels = zip(*batch)

    max_text_len = max(t.shape[0] for t in texts)
    padded_texts = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, :t.shape[0]] = t

    max_mel_len = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_mel_len)
    for i, mel in enumerate(mels):
        padded_mels[i, :, :mel.shape[1]] = mel

    return padded_texts, padded_mels

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

if __name__ == "__main__":
    lj_speech = LJSpeechDataset("LJSpeech-1.1/metadata.csv", "LJSpeech-1.1/wavs")

