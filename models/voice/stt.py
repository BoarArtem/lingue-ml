import whisper
from faster_whisper import WhisperModel
import uuid
import requests
import os
import torch
import pathlib

# Walk up from this file to the project root directory named "linguo-ml".
# Falls back to three levels up (models/voice/stt.py -> linguo-ml) if not found.
_PARENTS = pathlib.Path(__file__).resolve().parents
PROJECT_DIR = next((p for p in _PARENTS if p.name == "linguo-ml"), _PARENTS[2])
TEMPORARY_DIR = os.path.join(PROJECT_DIR, "temporary")

class Whisper():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.whisper_model = whisper.load_model(name="medium", device=self.device)

    def get_text(self, wav_pth: str):
        result = self.whisper_model.transcribe(audio=wav_pth)

        return result["text"]


class FasterWhisper(Whisper):
    def __init__(self):
        super().__init__()
        # CTranslate2 (faster-whisper's backend) supports only CUDA or CPU --
        # no Apple MPS. float16 is CUDA-only, so fall back to int8 on CPU.
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"

        self.faster_whisper_model = WhisperModel("medium", device=device, compute_type=compute_type)

    def get_text_faster(self, wav_pth: str):
        """
        :param wav_pth: Файл .wav формата
        :return: Возвращает готовый текст из аудио, где язык подбирается автоматически на основе озвучки
        """
        sigments, info = self.faster_whisper_model.transcribe(wav_pth)

        for sigment in sigments:
            return sigment.text

    def get_text_through_link(self, wav_link: str):
        """
        :param wav_link: Ссылка на .wav файл
        :return: Возвращает готовый текст из аудио, где язык подбирается автоматически на основе озвучки
        """
        # Some servers block the default "python-requests" User-Agent (HTTP 406).
        # Any non-default value works -- no browser spoofing needed.
        headers = {"User-Agent": "linguo-ml"}
        try:
            resp = requests.get(wav_link, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.content
        except requests.exceptions.RequestException as e:
            print(e)
            return None

        temp_file_name = uuid.uuid4().hex + ".wav"
        temp_file_path = os.path.join(TEMPORARY_DIR, temp_file_name)

        # The downloaded content is already a complete .wav file, so write the
        # raw bytes straight to disk -- no re-encoding needed.
        os.makedirs(TEMPORARY_DIR, exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(data)

        response = self.get_text(temp_file_path)

        os.remove(temp_file_path)
        print(temp_file_path)

        return response

if __name__ == "__main__":
    f_whisper = FasterWhisper()

    response = f_whisper.get_text_through_link("https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0018_8k.wav")

    print(response)

