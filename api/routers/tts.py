import io
import base64
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..dependencies import get_tts
from ..logger import build_logger
from ..schemas import TTSRequest, TTSMelResponse

router = APIRouter(prefix="/tts", tags=["TTS"])
limiter = Limiter(key_func=get_remote_address)
logger = build_logger("ml_linguo")


@router.post("/speech", summary="Синтез речи — возвращает WAV аудио")
@limiter.limit("5/minute")
def tts_speech(request: Request, req: TTSRequest, tts=Depends(get_tts)):
    rid = getattr(request.state, "request_id", "-")
    logger.info(f"TTS speech: chars={len(req.text)}", extra={"request_id": rid})

    try:
        audio = tts.generate(
            text=req.text,
            ref_audio=req.ref_audio_path,
            ref_text=req.ref_text,
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
@limiter.limit("5/minute")
def tts_mel(request: Request, req: TTSRequest, tts=Depends(get_tts)):
    rid = getattr(request.state, "request_id", "-")
    logger.info(f"TTS mel: chars={len(req.text)}", extra={"request_id": rid})

    try:
        audio = tts.generate(
            text=req.text,
            ref_audio=req.ref_audio_path,
            ref_text=req.ref_text,
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
