from models.llm.llm_live_talk import LLMWithTools
from models.voice.stt import FasterWhisper
from inference.omnivoice_tts_inference import OmniVoiceInference

import requests
import io
from scipy.io.wavfile import read

class Avatar:
    def __init__(self):
        self.stt = FasterWhisper()
        self.llm = LLMWithTools()
        self.tts = OmniVoiceInference()

    def invoke(self, audio_link: str):
        try:
            audio = requests.get(audio_link).content
            audio = io.BytesIO(audio)
        except Exception as e:
            print(e)

        try:
            response = self.stt.get_text(audio_link)

