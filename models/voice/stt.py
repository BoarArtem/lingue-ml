import whisper
from faster_whisper import WhisperModel
import torch

class Whisper():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.whisper_model = whisper.load_model(name="medium", device=self.device)

    def get_text(self, wav_pth: str):
        result = self.whisper_model.transcribe(audio=wav_pth)

        return result["text"]


class FasterWhisper():
    def __init__(self):
        self.faster_whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")

    def get_text(self, wav_pth: str):
        """
        :param wav_pth: Файл .wav формата
        :return: Возвращает готовый текст из аудио, где язык подбирается автоматически на основе озвучки
        """
        sigments, info = self.faster_whisper_model.transcribe(wav_pth)

        for sigment in sigments:
            return sigment.text
