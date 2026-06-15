import io
import base64
from pathlib import Path
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..dependencies import get_tts
from ..logger import build_logger
from ..schemas import TTSRequest, TTSMelResponse, MelToSpeechRequest

router = APIRouter(prefix="/tts", tags=["TTS"])
limiter = Limiter(key_func=get_remote_address)
logger = build_logger("ml_linguo")

# Fixed reference voice used for all TTS requests.
REF_AUDIO_PATH = str(Path(__file__).resolve().parents[2] / "inference" / "ref.wav")
REF_TEXT = "Існують, звичайно, християнські теологічні пояснення цієї традиції, але це цілком може бути дохристиянський ритуал весни та родючості."


@router.post("/speech", summary="Синтез речи — возвращает WAV аудио")
@limiter.limit("15/minute")
def tts_speech(request: Request, req: TTSRequest, tts=Depends(get_tts)):
    """
    Синтезирует речь из текста и возвращает готовый WAV-файл.
    Использует фиксированную референсную речь и аудио (REF_TEXT / REF_AUDIO_PATH),
    чтобы обеспечить единообразие тембра и качества между запросами.
    Поддерживает указание языка синтеза; если не указан — берётся язык модели по умолчанию.

    Limits:
        Не более 15 запросов в минуту с одного IP.

    Args:
        req (TTSRequest):
            - text (str): Текст для синтеза речи.
            - language (str, optional): Код языка синтеза. Если None — используется язык модели по умолчанию.

    Returns:
        StreamingResponse: Аудиопоток в формате audio/wav.
    """
    rid = getattr(request.state, "request_id", "-")
    logger.info(f"TTS speech: chars={len(req.text)}", extra={"request_id": rid})

    try:
        audio = tts.generate(
            text=req.text,
            ref_audio=REF_AUDIO_PATH,
            ref_text=REF_TEXT,
            **({} if req.language is None else {"language": req.language}),
        )
    except Exception:
        logger.error("TTS generation failed", exc_info=True, extra={"request_id": rid})
        raise HTTPException(500, "TTS generation failed")

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=tts.sample_rate, format="wav")
    buf.seek(0)

    logger.info(f"TTS speech done: duration={len(audio)/tts.sample_rate:.2f}s",
                extra={"request_id": rid})
    return StreamingResponse(buf, media_type="audio/wav")


@router.post("/mel", summary="Синтез речи — возвращает мел-спектрограмму", response_model=TTSMelResponse)
@limiter.limit("15/minute")
def tts_mel(request: Request, req: TTSRequest, tts=Depends(get_tts)):
    """
    Синтезирует речь из текста и возвращает не WAV, а её мел-спектрограмму.
    Сначала генерирует аудио (как /speech), затем считает мел-спектрограмму
    с заданными параметрами FFT. Сам массив отдаётся как float32 в Base64.

    Limits:
        Не более 15 запросов в минуту с одного IP.

    Args:
        req (TTSRequest):
            - text (str): Текст для синтеза речи.
            - language (str, optional): Код языка синтеза. Если None — язык модели по умолчанию.
            - n_fft (int): Размер окна FFT. По умолчанию 1024.
            - hop_length (int): Шаг окна (hop length). По умолчанию 256.
            - n_mels (int): Количество мел-фильтров. По умолчанию 80.

    Returns:
        TTSMelResponse: shape [n_mels, временные шаги], данные float32 в Base64
        и параметры (sample_rate, n_mels, n_fft, hop_length) для восстановления.
    """
    rid = getattr(request.state, "request_id", "-")
    logger.info(f"TTS mel: chars={len(req.text)}", extra={"request_id": rid})

    try:
        audio = tts.generate(
            text=req.text,
            ref_audio=REF_AUDIO_PATH,
            ref_text=REF_TEXT,
            **({} if req.language is None else {"language": req.language}),
        )
        mel = tts.mel_spectrogram(
            audio,
            n_fft=req.n_fft,
            hop_length=req.hop_length,
            n_mels=req.n_mels,
        )
    except Exception:
        logger.error("TTS mel generation failed", exc_info=True, extra={"request_id": rid})
        raise HTTPException(500, "TTS mel generation failed")

    data = base64.b64encode(mel.astype(np.float32).tobytes()).decode("utf-8")

    logger.info(f"TTS mel done: shape={list(mel.shape)}", extra={"request_id": rid})
    return TTSMelResponse(
        shape=list(mel.shape),
        data=data,
        sample_rate=tts.sample_rate,
        n_mels=req.n_mels,
        n_fft=req.n_fft,
        hop_length=req.hop_length,
    )


@router.post("/mel/speech", summary="Восстановление аудио из мел-спектрограммы — возвращает WAV")
@limiter.limit("15/minute")
def tts_mel_to_speech(request: Request, req: MelToSpeechRequest, tts=Depends(get_tts)):
    """
    Обратная операция к /mel: восстанавливает аудио из мел-спектрограммы и возвращает WAV.
    Декодирует Base64-массив float32, восстанавливает форму через reshape по shape
    и прогоняет через алгоритм Griffin-Lim для получения сигнала.

    Параметры n_fft и hop_length должны совпадать с теми, что использовались при
    создании мел-спектрограммы, иначе результат будет искажён.

    Limits:
        Не более 15 запросов в минуту с одного IP.

    Args:
        req (MelToSpeechRequest):
            - shape (list[int]): Форма массива [n_mels, временные шаги].
            - data (str): Массив мел-спектрограммы float32 в Base64.
            - n_fft (int): Размер окна FFT. Должен совпадать с использованным при создании. По умолчанию 1024.
            - hop_length (int): Шаг окна. Должен совпадать с использованным при создании. По умолчанию 256.
            - n_iter (int): Количество итераций Griffin-Lim — больше итераций медленнее, но чище звук. По умолчанию 32.

    Returns:
        StreamingResponse: Восстановленный аудиопоток в формате audio/wav.
    """
    rid = getattr(request.state, "request_id", "-")
    logger.info(f"TTS mel->speech: shape={req.shape}", extra={"request_id": rid})

    try:
        mel = np.frombuffer(base64.b64decode(req.data), dtype=np.float32).reshape(req.shape)
        audio = tts.mel_to_audio(
            mel,
            n_fft=req.n_fft,
            hop_length=req.hop_length,
            n_iter=req.n_iter,
        )
    except Exception:
        logger.error("TTS mel->speech failed", exc_info=True, extra={"request_id": rid})
        raise HTTPException(500, "TTS mel->speech failed")

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=tts.sample_rate, format="wav")
    buf.seek(0)

    logger.info(f"TTS mel->speech done: duration={len(audio)/tts.sample_rate:.2f}s",
                extra={"request_id": rid})
    return StreamingResponse(buf, media_type="audio/wav")
