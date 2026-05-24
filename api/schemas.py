from pydantic import BaseModel, Field
from typing import Any


class SimilarRequest(BaseModel):
    arr: list[str] = Field(example=["dog", "cat"])
    topn: int = Field(default=10, example=5)

class SimilarItem(BaseModel):
    word: str = Field(example="puppy")
    score: float = Field(example=0.91)

class SimilarResponse(BaseModel):
    results: list[SimilarItem]


class Features(BaseModel):
    unique_words: int = Field(
        example=1500,
        description="Количество уникальных изученных слов"
    )

    words_a1: int = Field(
        example=600,
        description="Количество слов уровня A1"
    )

    words_a2: int = Field(
        example=500,
        description="Количество слов уровня A2"
    )

    words_b1: int = Field(
        example=400,
        description="Количество слов уровня B1"
    )

    words_b2: int = Field(
        example=0,
        description="Количество слов уровня B2"
    )

    avg_acc_7d: float = Field(
        example=0.88,
        description="Средняя точность ответов за последние 7 дней"
    )

    avg_acc_30d: float = Field(
        example=0.85,
        description="Средняя точность ответов за последние 30 дней"
    )

    avg_time_sec: float = Field(
        example=6.0,
        description="Среднее время ответа пользователя в секундах"
    )

    words_day_7d: int = Field(
        example=30,
        description="Количество изученных слов за последние 7 дней"
    )

    words_day_30d: int = Field(
        example=900,
        description="Количество изученных слов за последние 30 дней"
    )

    streak: int = Field(
        example=20,
        description="Количество дней подряд активности"
    )

    sessions_week: int = Field(
        example=14,
        description="Количество учебных сессий за неделю"
    )


class PredictRequest(BaseModel):
    features: Features


class PredictResponse(BaseModel):
    prediction: int = Field(
        example=120,
        description="Количество дней до достижения уровня B2"
    )


class WordLevelRequest(BaseModel):
    word: str = Field(example="nevertheless")
    translation: str = Field(example="тем не менее")

class WordLevelResponse(BaseModel):
    level: str = Field(example="C1", description="CEFR уровень: A1 A2 B1 B2 C1 C2")


class SentenceRequest(BaseModel):
    word: str = Field(example="dog")
    level: str = Field(example="A1", description="CEFR уровень")
    language: str = Field(example="en", description="Код языка: en | ru | es | fr | de | ch")

class SentenceResponse(BaseModel):
    sentence: str = Field(example="The dog is big.")


class PreprocessRequest(BaseModel):
    sentence: str = Field(example="Dogs are running in the park")
    language: str = Field(example="en", description="en | ru | es | fr | de | ch")

class PreprocessResponse(BaseModel):
    tokens: list[str] = Field(example=["dog", "run", "park"])


class TTSRequest(BaseModel):
    text: str = Field(example="Hello, this is a test.")
    ref_audio_path: str = Field(example="/inference/ref.wav", description="Server-side path to reference WAV file")
    ref_text: str = Field(example="Transcription of the reference audio.")
    language: str | None = Field(default=None, example="en")
    n_fft: int = Field(default=1024, description="FFT size for mel spectrogram")
    hop_length: int = Field(default=256, description="Hop length for mel spectrogram")
    n_mels: int = Field(default=80, description="Number of mel filterbanks")


class TTSSpeechResponse(BaseModel):
    sample_rate: int = Field(example=24000)
    duration_sec: float = Field(example=3.5)


class TTSMelResponse(BaseModel):
    shape: list[int] = Field(example=[80, 383], description="[n_mels, time_steps]")
    data: str = Field(description="Base64-encoded float32 numpy array, reshape using shape")
    sample_rate: int = Field(example=24000)
    n_mels: int = Field(example=80)
    n_fft: int = Field(example=1024)
    hop_length: int = Field(example=256)


class ModelsStatus(BaseModel):
    word2vec: bool
    b2: bool
    tts: bool

class HealthResponse(BaseModel):
    status: str = Field(example="ok", description="ok | degraded")
    version: str = Field(example="v2.11.5")
    models: ModelsStatus