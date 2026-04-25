from models import load_tacotron2
import torch

tacotron2 = load_tacotron2("../models/tacotron2_epoch_1.pth")
tacotron2.eval()

vocoder = None
vocoder.eval()

def decode_mel(mel):
    # mel: [B, T_dec, 80]
    mel_out = mel.transpose(1, 2)

    with torch.no_grad():
        audio = vocoder.inference(mel_out.unsqueeze(0))[0]

    return audio

def get_mel(text):
    with torch.no_grad():
        _, mel_outs, _, _, _ = tacotron2(text)
    return mel_outs.squeeze(0)

def mel_to_audio(mel):
    with torch.no_grad():
        audio = vocoder(mel)
    return audio

def tts(text):
    mel = get_mel(text)
    audio = mel_to_audio(mel)

    return audio

if __name__ == "__main__":
    print(tts("Hello"))