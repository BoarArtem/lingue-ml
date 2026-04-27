import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import load_tacotron2
from data.ljspeech.dataset_executor import text_to_sequence
import torch

MODEL_PATH = str(Path(__file__).parent.parent / "tacotron2_epoch_11.pth")

tacotron2 = load_tacotron2(MODEL_PATH)
tacotron2.eval()

_model_device = next(tacotron2.parameters()).device

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'map_location': 'cpu'})
try:
    vocoder = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
finally:
    torch.load = _original_load
vocoder = vocoder.remove_weightnorm(vocoder)
vocoder.eval()
vocoder = vocoder.to(_model_device)

def text_to_tensor(text):
    indices = text_to_sequence(text)
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(_model_device)

def get_mel(text, max_steps=500, stop_threshold=0.3):
    with torch.no_grad():
        x = text_to_tensor(text)
        _, mel_outs, _, _, _ = tacotron2(x, max_steps=max_steps, stop_threshold=stop_threshold)
    # mel_outs: [1, T, 80] → squeeze → [T, 80] → transpose → [80, T]
    return mel_outs.squeeze(0).transpose(0, 1)

def mel_to_audio(mel):
    # WaveGlow expects [batch, 80, T]
    with torch.no_grad():
        audio = vocoder.infer(mel.unsqueeze(0))
    return audio.squeeze(0)

def tts(text):
    mel = get_mel(text)
    audio = mel_to_audio(mel)
    return audio

if __name__ == "__main__":
    import numpy as np
    import scipy.io.wavfile as wav
    audio = tts("You look so pretty")
    audio_np = audio.cpu().numpy().astype(np.float32)
    wav.write("output.wav", 22050, audio_np)
    print("Saved output.wav")
