import time
import pandas as pd
from fastapi import APIRouter, Depends, Request
from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from inference.spam_classification_inference import spam_or_ham
from ..dependencies import get_spam_model, get_b2_model, get_topic_predictor, get_anti_plagiarism, get_word2vec, \
    get_spam_vocab, get_device
from ..logger import build_logger

from ..schemas import (
    SpamClassificationRequest, SpamClassificationResponse,
    PredictRequest, PredictResponse,
    SingleTopicRequest, TopicResponse,
    TopicRequest, TopicsResponse,
    CheckPlagiarismRequest, CheckPlagiarismResponse, SimilarResponse, SimilarRequest, SimilarItem,
)

router = APIRouter(tags=["Machine Learning"])
limiter = Limiter(key_func=get_remote_address)
logger = build_logger("ml_linguo")


@router.post(
    "/spam_classification",
    summary="Классификация карточки на спам",
    response_model=SpamClassificationResponse,
)
@limiter.limit("30/minute")
def spam_classification(
    request: Request,
    req: SpamClassificationRequest,
    spam_model=Depends(get_spam_model),
    spam_vocab=Depends(get_spam_vocab),
    device=Depends(get_device),
):
    """
    Классифицирует текст карточки как спам (spam) или нормальный (ham).
    Использует обученную модель спам-классификации и её словарь.

    Limits:
        Не более 30 запросов в минуту с одного IP.

    Args:
        req (SpamClassificationRequest):
            - user_sentence (str): Текст карточки для классификации.

    Returns:
        SpamClassificationResponse: label (str) — "spam" или "ham".
    """
    rid = getattr(request.state, "request_id", "-")
    logger.info(f"Spam classification: chars={len(req.user_sentence)}",
                extra={"request_id": rid})
    try:
        result = spam_or_ham(req.user_sentence, spam_model, spam_vocab, device)
    except Exception:
        logger.error("Spam classification error", exc_info=True,
                     extra={"request_id": rid})
        raise HTTPException(500, "Spam classification error")

    logger.info(f"Spam result: '{result}'",
                extra={"request_id": rid, "prediction": result})

    return SpamClassificationResponse(label=result)

@router.post("/similar", response_model=SimilarResponse)
@limiter.limit("30/minute")
def similar(request: Request, req: SimilarRequest, wv=Depends(get_word2vec)):
    """
    Находит наиболее похожие слова по эмбеддингам модели Word2Vec.
    Принимает список слов и возвращает topn ближайших по косинусной близости.

    Limits:
        Не более 30 запросов в минуту с одного IP.

    Args:
        req (SimilarRequest):
            - arr (list[str]): Слова, для которых ищутся похожие.
            - topn (int): Сколько похожих слов вернуть. По умолчанию 10.

    Returns:
        SimilarResponse: results — список пар слово/оценка близости (score).

    Raises:
        HTTPException 404: если хотя бы одного слова нет в словаре модели.
    """
    try:
        raw = wv.wv.most_similar(req.arr, topn=req.topn)
    except KeyError as e:
        raise HTTPException(404, f"Word not found: {e}")
    return SimilarResponse(
        results=[SimilarItem(word=word, score=score) for word, score in raw]
    )


@router.post(
    "/predict",
    summary="ML предсказание B2",
    response_model=PredictResponse,
)
def predict(
    request: Request,
    req: PredictRequest,
    b2=Depends(get_b2_model),
):
    """
    Предсказывает количество дней до достижения уровня B2 по статистике ученика.
    Признаки пользователя превращаются в DataFrame и подаются в обученную модель.

    Args:
        req (PredictRequest):
            - features (Features): Набор признаков обучения (изученные слова,
              средняя точность, стрик, количество сессий и т.д.).

    Returns:
        PredictResponse: prediction (int) — прогноз количества дней до уровня B2.

    Raises:
        HTTPException 400: если модель не обучена или отсутствуют нужные колонки признаков.
    """
    rid = getattr(request.state, "request_id", "-")

    if not b2.feature_names:
        raise HTTPException(400, "Модель не обучена")

    df = pd.DataFrame([req.features.model_dump()])
    missing = [c for c in b2.feature_names if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Отсутствуют колонки: {missing}")

    pred = b2.model.predict(df[b2.feature_names])[0]
    return PredictResponse(prediction=int(pred))


@router.post(
    "/predict_topic",
    summary="Определение темы — одно предложение",
    response_model=TopicResponse,
)
@limiter.limit("10/minute")
def predict_topic(
    request: Request,
    req: SingleTopicRequest,
    topic=Depends(get_topic_predictor),
):
    """
    Определяет тему одного предложения.

    Limits:
        Не более 10 запросов в минуту с одного IP.

    Args:
        req (SingleTopicRequest):
            - sentence (str): Предложение, тему которого нужно определить.

    Returns:
        TopicResponse: topic (str) — предсказанная тема.
    """
    rid = getattr(request.state, "request_id", "-")
    result = topic.get_topic(req.sentence)
    logger.info(f"Topic: '{result}'", extra={"request_id": rid, "topic": result})
    return TopicResponse(topic=result)


@router.post(
    "/predict_topics",
    summary="Определение тем — массив предложений",
    response_model=TopicsResponse,
)
@limiter.limit("10/minute")
def predict_topics(
    request: Request,
    req: TopicRequest,
    topic=Depends(get_topic_predictor),
):
    """
    Определяет темы сразу для массива предложений (пакетная обработка).

    Limits:
        Не более 10 запросов в минуту с одного IP.

    Args:
        req (TopicRequest):
            - sentences (list[str]): Список предложений для определения тем.

    Returns:
        TopicsResponse: topics (list[str]) — темы в порядке входных предложений.
    """
    results = topic.get_topics(req.sentences)
    return TopicsResponse(topics=results)


@router.post(
    "/check_plagiarism",
    summary="Проверка текста на AI-плагиат",
    response_model=CheckPlagiarismResponse,
)
@limiter.limit("10/minute")
def check_plagiarism(
    request: Request,
    req: CheckPlagiarismRequest,
    anti_plag=Depends(get_anti_plagiarism),
):
    """
    Проверяет, написан ли текст человеком или сгенерирован ИИ (AI-плагиат).
    По умолчанию возвращает строковую метку; при get_index=True — числовой индекс.

    Limits:
        Не более 10 запросов в минуту с одного IP.

    Args:
        req (CheckPlagiarismRequest):
            - user_text (str): Текст для проверки на AI-плагиат.
            - get_index (bool): Если True — вернуть числовой индекс вместо строки. По умолчанию False.

    Returns:
        CheckPlagiarismResponse: label — "human" | "ai" (или 0 | 1 при get_index=True).
    """
    label = anti_plag.get_label(req.user_text)

    if req.get_index:
        label = anti_plag.get_index_from_label(label)

    return CheckPlagiarismResponse(label=label)