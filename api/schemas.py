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


class SpamClassificationRequest(BaseModel):
    user_sentence: str = Field(example="buy cheap drugs now")

class SpamClassificationResponse(BaseModel):
    label: str = Field(example="ham", description="spam | ham")


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


class SingleTopicRequest(BaseModel):
    sentence: str = Field(example="I love coding in Python")

class TopicResponse(BaseModel):
    topic: str = Field(example="Technology")


class TopicRequest(BaseModel):
    sentences: list[str] = Field(example=["I love coding", "I like football"])

class TopicsResponse(BaseModel):
    topics: list[str] = Field(example=["Technology", "Sport"])


class CheckPlagiarismRequest(BaseModel):
    user_text: str = Field(example="The quick brown fox jumps over the lazy dog")
    get_index: bool = Field(default=False, description="Вернуть числовой индекс вместо строки")

class CheckPlagiarismResponse(BaseModel):
    label: str | int = Field(example="human", description="human | ai или 0 | 1 если get_index=True")


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


class CorrectParagraphRequest(BaseModel):
    user_sentence: str = Field(example="I eated pizza yesterday")

class WordChange(BaseModel):
    incorrect: str = Field(example="eated")
    correct: str = Field(example="ate")

class CorrectParagraphResponse(BaseModel):
    user_sentence: str = Field(example="I eated pizza yesterday")
    ai_sentence: str = Field(example="I ate pizza yesterday")
    changes: list[WordChange]


class PreprocessRequest(BaseModel):
    sentence: str = Field(example="Dogs are running in the park")
    language: str = Field(example="en", description="en | ru | es | fr | de | ch")

class PreprocessResponse(BaseModel):
    tokens: list[str] = Field(example=["dog", "run", "park"])


class TTSRequest(BaseModel):
    text: str = Field(example="Hello, this is a test.")
    language: str | None = Field(default=None, example="en")
    n_fft: int = Field(default=1024, description="Размер FFT для мел-спектрограммы")
    hop_length: int = Field(default=256, description="Шаг окна (hop length) для мел-спектрограммы")
    n_mels: int = Field(default=80, description="Количество мел-фильтров")


class TTSSpeechResponse(BaseModel):
    sample_rate: int = Field(example=24000)
    duration_sec: float = Field(example=3.5)


class MelToSpeechRequest(BaseModel):
    shape: list[int] = Field(example=[80, 173], description="[n_mels, кол-во временных шагов] массива мел-спектрограммы")
    data: str = Field(description="Массив мел-спектрограммы float32 в Base64 (по строкам, в соответствии с shape)")
    n_fft: int = Field(default=1024, description="Должен совпадать с n_fft, использованным при создании мел-спектрограммы")
    hop_length: int = Field(default=256, description="Должен совпадать с hop_length, использованным при создании мел-спектрограммы")
    n_iter: int = Field(default=32, description="Количество итераций Griffin-Lim — больше итераций медленнее, но чище звук")


class TTSMelResponse(BaseModel):
    shape: list[int] = Field(example=[80, 383], description="[n_mels, кол-во временных шагов]")
    data: str = Field(description="Массив numpy float32 в Base64, восстанавливается через reshape по shape")
    sample_rate: int = Field(example=24000)
    n_mels: int = Field(example=80)
    n_fft: int = Field(example=1024)
    hop_length: int = Field(example=256)


class ModelsStatus(BaseModel):
    word2vec: bool
    spam: bool
    b2: bool
    topic: bool
    anti_plagiarism: bool
    tts: bool

class HealthResponse(BaseModel):
    status: str = Field(example="ok", description="ok | degraded")
    version: str = Field(example="v2.11.5")
    models: ModelsStatus
