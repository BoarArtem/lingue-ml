from omnivoice import OmniVoice
import soundfile as sf
import torch
import torchaudio
import numpy as np


class OmniVoiceInference:
    def __init__(self, model_name: str = "k2-fsa/OmniVoice", device: str = "cpu", dtype=torch.float32):
        self.model = OmniVoice.from_pretrained(model_name, device_map=device, dtype=dtype)
        self.sample_rate: int = self.model.sampling_rate

    def generate(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        **kwargs,
    ) -> np.ndarray:
        results = self.model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, **kwargs)
        return results[0]

    def save(self, audio: np.ndarray, path: str) -> None:
        sf.write(path, audio, samplerate=self.sample_rate)

    def mel_spectrogram(
        self,
        audio: np.ndarray,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ) -> np.ndarray:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel = transform(waveform)
        return mel.squeeze(0).numpy()

    def mel_to_audio(
        self,
        mel: np.ndarray,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_iter: int = 32,
    ) -> np.ndarray:
        """Reconstruct a waveform from a power mel-spectrogram via Griffin-Lim.

        Inverts the same transform produced by ``mel_spectrogram`` (an un-logged
        power spectrogram). Quality is approximate — Griffin-Lim has no learned
        phase — but needs no extra model.
        """
        mel_t = torch.from_numpy(np.ascontiguousarray(mel)).float()
        n_mels = mel_t.shape[0]

        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=self.sample_rate,
        )
        spectrogram = inverse_mel(mel_t)

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=n_iter,
        )
        waveform = griffin_lim(spectrogram)
        return waveform.numpy()

    def generate_mel(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        **kwargs,
    ) -> np.ndarray:
        audio = self.generate(text, ref_audio, ref_text, **kwargs)
        return self.mel_spectrogram(audio, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


if __name__ == "__main__":
    tts = OmniVoiceInference()

    audio = tts.generate(
        text="Привет всем, это тестирование модели OmniVoice, жаль, но Tacotron 2 Русику в падлу обучать и дорого, поэтому любуемся взятой моделью, оцените и напишите честный отзыв",
        ref_audio="ref.wav",
        ref_text="Існують, звичайно, християнські теологічні пояснення цієї традиції, але це цілком може бути дохристиянський ритуал весни та родючості.",
    )
    tts.save(audio, "output.wav")

    mel = tts.mel_spectrogram(audio)
    print(f"audio shape: {audio.shape}, mel shape: {mel.shape}")
