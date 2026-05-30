import asyncio
import time

from fastapi import APIRouter, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..logger import build_logger
from ..schemas import (
    WordLevelRequest, WordLevelResponse,
    SentenceRequest, SentenceResponse,
    CorrectParagraphRequest, CorrectParagraphResponse, WordChange,
)
from models import llm_word_level, llm_sentence_generate, correct_paragraph, get_changed_word, word_pair

router = APIRouter(tags=["LLM"])
limiter = Limiter(key_func=get_remote_address)
logger = build_logger("ml_linguo")

@router.post(
    "/word_level",
    summary="Определить уровень слова CEFR",
    response_model=WordLevelResponse,
)
def word_level(request: Request, req: WordLevelRequest):
    rid = getattr(request.state, "request_id", "-")

    logger.info(
        f"Word level request: word='{req.word}'",
        extra={"request_id": rid, "word": req.word}
    )
    t0 = time.perf_counter()

    try:
        result = llm_word_level(req.word, req.translation)
    except Exception:
        logger.error(
            f"Word level failed: word='{req.word}'",
            exc_info=True,
            extra={"request_id": rid, "word": req.word}
        )
        raise HTTPException(status_code=500, detail="LLM error")

    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        f"Word level done: word='{req.word}' → '{result}', took={ms}ms",
        extra={"request_id": rid, "word": req.word, "duration_ms": ms}
    )

    return WordLevelResponse(level=result)


@router.post(
    "/sentence",
    summary="Сгенерировать предложение",
    response_model=SentenceResponse,
)
@limiter.limit("10/minute")
def sentence(request: Request, req: SentenceRequest):
    rid = getattr(request.state, "request_id", "-")

    logger.info(
        f"Sentence generation: word='{req.word}', level={req.level}, lang={req.language}",
        extra={"request_id": rid, "word": req.word, "language": req.language}
    )
    t0 = time.perf_counter()

    try:
        result = llm_sentence_generate(req.word, req.level, req.language)
    except Exception:
        logger.error(
            f"Sentence generation failed: word='{req.word}'",
            exc_info=True,
            extra={"request_id": rid, "word": req.word}
        )
        raise HTTPException(status_code=500, detail="LLM error")

    ms = round((time.perf_counter() - t0) * 1000, 1)
    preview = result[:60] + ("…" if len(result) > 60 else "")
    logger.info(
        f"Sentence generated: '{preview}', took={ms}ms",
        extra={"request_id": rid, "word": req.word, "duration_ms": ms}
    )

    return SentenceResponse(sentence=result)


@router.post(
    "/correct_paragraph",
    summary="Исправление ошибок в предложении",
    response_model=CorrectParagraphResponse,
)
@limiter.limit("10/minute")
async def correct_paragraph_checking(request: Request, req: CorrectParagraphRequest):
    rid = getattr(request.state, "request_id", "-")

    logger.info(
        f"Correct paragraph: chars={len(req.user_sentence)}",
        extra={"request_id": rid}
    )
    t0 = time.perf_counter()

    try:
        ai_sentence = await asyncio.to_thread(correct_paragraph, req.user_sentence)
    except Exception:
        logger.error(
            "Correct paragraph LLM call failed",
            exc_info=True,
            extra={"request_id": rid}
        )
        raise HTTPException(status_code=500, detail="LLM error")

    try:
        incorrect, correct = await asyncio.to_thread(
            get_changed_word, req.user_sentence, ai_sentence
        )
    except Exception:
        logger.error(
            "get_changed_word failed",
            exc_info=True,
            extra={"request_id": rid}
        )
        raise HTTPException(status_code=500, detail="Diff error")

    changes = [
        WordChange(incorrect=inc, correct=cor)
        for inc, cor in word_pair(incorrect, correct)
    ]

    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        f"Correct paragraph done: changes={len(changes)}, took={ms}ms",
        extra={"request_id": rid, "duration_ms": ms}
    )

    return CorrectParagraphResponse(
        user_sentence=req.user_sentence,
        ai_sentence=ai_sentence,
        changes=changes,
    )