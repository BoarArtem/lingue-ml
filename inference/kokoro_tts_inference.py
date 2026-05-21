from openai import OpenAI
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random
import io
import os

client = OpenAI(
    base_url="http://localhost:8880/v1",
    api_key=os.getenv("OPENAI_KEY", "not-needed")
)

@dataclass
class KokoroTTSParams:
    model: str = "kokoro"
    voice: str = "af_sky+af_bella"

class KokoroTTS:
    def __init__(self):
        self.client = client
        self.params = KokoroTTSParams()

    def generate_audio(self, prompt):

        id = random.randint(1, 1000000)

        with client.audio.speech.with_streaming_response.create(
            model=self.params.model,
            voice=self.params.voice, # single or multiple voicepack combo
            input=prompt,
            response_format="wav"
        ) as response:
            response.stream_to_file(f"output_{id}.wav")

        return f"output_{id}.wav", id

    def get_spectogram(self, prompt) -> np.ndarray:
        # collect audio bytes in memory
        buf = io.BytesIO()
        with client.audio.speech.with_streaming_response.create(
            model=self.params.model,
            voice=self.params.voice,
            input=prompt,
            response_format="wav"
        ) as response:
            for chunk in response.iter_bytes():
                buf.write(chunk)
        buf.seek(0)

        y, sr = librosa.load(buf, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        return S_db, sr

if __name__ == "__main__":

    tts = KokoroTTS()
    filename, _ = tts.generate_audio(
        "Я модель имплементирована Русланом Подолян! Оцените качество моей озвучки"
    )
    print(filename)