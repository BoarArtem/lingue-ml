import whisper
import torch

class WhisperModel():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.whisper_model = whisper.load_model(name="medium", device=self.device)

    def inference(self, wav_pth: str):
        result = self.whisper_model.transcribe(audio=wav_pth, language="en")

        return result["text"]
