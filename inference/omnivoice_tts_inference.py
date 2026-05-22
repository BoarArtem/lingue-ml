from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cpu",
    dtype=torch.float32
)

# Generate audio
audio = model.generate(
    text="Артем Бояр Бояяр Боояр Бояярр Бояяярррр Артемий Артемушка Артемуха",
    ref_audio="ref.wav",
    ref_text="Існують, звичайно, християнські теологічні пояснення цієї традиції, але це цілком може бути дохристиянський ритуал весни та родючості."
)

sf.write("output.wav", audio[0], samplerate=24000)