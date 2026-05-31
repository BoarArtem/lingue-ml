import io
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kokoro import KPipeline

app = FastAPI()

pipeline = KPipeline(lang_code="a")  # 'a' = American English


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "af_sky"
    response_format: str = "mp3"
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    voice = req.voice.split("+")[0]  # use first voice if combo given

    audio_chunks = []
    sample_rate = None
    for _, _, audio in pipeline(req.input, voice=voice, speed=req.speed):
        audio_chunks.append(audio)
        if sample_rate is None:
            sample_rate = 24000

    if not audio_chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    import numpy as np
    combined = np.concatenate(audio_chunks)

    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="wav")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")
