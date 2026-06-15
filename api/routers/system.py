from fastapi import APIRouter, Request
from ..config import settings
from ..schemas import HealthResponse

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    summary="Проверка состояния сервиса",
    response_model=HealthResponse,
)
def health(request: Request):
    """
    Проверка состояния сервиса (health check).
    Сообщает, какие модели загружены, и общий статус: "ok", если загружены все,
    иначе "degraded". Используется для мониторинга и проб готовности.

    Returns:
        HealthResponse: status ("ok" | "degraded"), version и флаги загрузки
        по каждой модели (word2vec, spam, b2, topic, anti_plagiarism, tts).
    """
    ml = request.app.state.ml

    models_status = {
        "word2vec":        ml.word2vec is not None,
        "spam":            ml.spam is not None,
        "b2":              ml.b2 is not None,
        "topic":           ml.topic is not None,
        "anti_plagiarism": ml.anti_plagiarism is not None,
        "tts":             ml.tts is not None,
    }

    all_ok = all(models_status.values())

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        version="None",
        models=models_status,
    )
