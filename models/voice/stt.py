import whisper
from faster_whisper import WhisperModel
import torch

class Whisper():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.whisper_model = whisper.load_model(name="medium", device=self.device)

    def get_text(self, wav_pth: str):
        result = self.whisper_model.transcribe(audio=wav_pth, language="en")

        return result["text"]


class FasterWhisper():
    def __init__(self):
        self.faster_whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")

    def get_text(self, wav_pth: str):
        sigments, info = self.faster_whisper_model.transcribe(wav_pth, language="ru")

        for sigment in sigments:
            return sigment.text


