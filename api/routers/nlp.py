from fastapi import Request
from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.logger import build_logger
from api.routers.ml import router
from api.schemas import PreprocessResponse, PreprocessRequest
from data.tokenizer import sentence_preprocess_english, sentence_preprocess_russian, sentence_preprocess_spanish, \
    sentence_preprocess_france, sentence_preprocess_german, sentence_preprocess_chinese

limiter = Limiter(key_func=get_remote_address)
logger = build_logger("ml_linguo")

PREPROCESSORS = {
    "en": sentence_preprocess_english,
    "ru": sentence_preprocess_russian,
    "es": sentence_preprocess_spanish,
    "fr": sentence_preprocess_france,
    "de": sentence_preprocess_german,
    "ch": sentence_preprocess_chinese,
}


@router.post(
    "/preprocess",
    summary="Предобработка предложения",
    response_model=PreprocessResponse,
)
def preprocess(request: Request, req: PreprocessRequest):
    """
    Предобрабатывает предложение: токенизация, нормализация, лемматизация и т.п.
    Конкретный препроцессор выбирается по коду языка из PREPROCESSORS.

    Args:
        req (PreprocessRequest):
            - sentence (str): Предложение для предобработки.
            - language (str): Код языка (en | ru | es | fr | de | ch).

    Returns:
        PreprocessResponse: tokens (list[str]) — список обработанных токенов.

    Raises:
        HTTPException 400: если язык не поддерживается.
    """
    rid = getattr(request.state, "request_id", "-")

    if req.language not in PREPROCESSORS:
        logger.warning(
            f"Unsupported language: '{req.language}'",
            extra={"request_id": rid, "language": req.language}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: '{req.language}'. Available: {list(PREPROCESSORS.keys())}"
        )

    logger.info(
        f"Preprocessing: lang={req.language}, chars={len(req.sentence)}",
        extra={"request_id": rid, "language": req.language}
    )

    tokens = PREPROCESSORS[req.language](req.sentence)

    logger.info(
        f"Preprocessed: tokens={len(tokens)}",
        extra={"request_id": rid}
    )

    return PreprocessResponse(tokens=tokens)