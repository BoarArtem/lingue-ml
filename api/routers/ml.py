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
    Модель которая классифицирует на спам ОДНУ карточку

    Limits:
        Не более 30 запросов в минуту с одного IP

    Args:
        req (SpamClassificationRequest):
            - user_sentence (str): Предложение пользователя

    Returns:
        Возвращает класс: spam | ham, где spam - спам, ham - нормалдасик

    Raises:
        500: Spam classification error
        422: Validation Error
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
    Модель которая из списка слов, представляет слова которые могут быть похожие по смыслу

    Limits:
        Не более 30 запросов в минуту с одного IP

    Args:
        req (SimilarRequest):
            - arr (list[str]): Список слов пользователя
            - topn (int): Количество похожих слов которая выдаст модель

    Returns:
        Возвращает массив диктов в котором каждый дикт состоит из похожего слова и процентом схожести(score)

    Raises:
        404: Word not found
        422: Validation Error
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
    Модель которая делает определяет за сколько пользователь дойдет до уровня B2

    Args:
        req (PredictRequest):
            - unique_words (int): Количество выученных уникальных слов
            - words_a1 (int): Количество выученных слов уровня A1
            - words_a2 (int): Количество выученных слов уровня А2
            - words_b1 (int): Количество выученных слов уровня B1
            - words_b2 (int): Количество выученных слов уровня B2
            - avg_acc_7d (float): Средняя точность ответов за последние 7 дней
            - avg_acc_30d (float): Средняя точность ответов за последние 30 дней
            - avg_time_sec (float): Среднее время ответа пользователя в секудах
            - words_day_7d (int): Количество выученных слов за последние 7 дней
            - words_day_30d (int): Количество выученных слов за последние 30 дней
            - streak (int): Количество дней подряд активности
            - session_week (int): Количество учебных сессий за неделю

    Returns:
        Возвращает число - дни за которые пользователь может дойти до уровня B2

    Raises:
        400: Модель еще не обучена
        400: Отсутствие каких либо колонок
        422: Validation Error
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
    Модель которая по предложению (ОДНОЙ КАРТОЧКИ) пользователя может определить его тему (tags - если в проекте)

    Limits:
        Не более 30 запросов в минуту с одного IP


    Args:
        req (SingleTopicRequest):
            - sentence (str): Предложение пользователя

    Returns:
        Возвращает соответствующий тег

    Raises:
        Пока отсутствуют (дима не сделал обработчики!!!!!)
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
    Модель которая по предложения (ПО ВСЕЙ КОЛОДЕ) пользователя может определить темы (tags - если в проекте)

    Limits:
        Не более 30 запросов в минуту с одного IP


    Args:
        req (TopicRequest):
            - sentences (list[str]): Список предложений

    Returns:
        Возвращает соответствующие теги для каждого предложени

    Raises:
        Пока отсутствуют (дима не сделал обработчики!!!!!)
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
    Модель которая по тексту (текст именно большой, не предложение) определяет ИИ писал или не ИИ

    Limits:
        Не более 10 запросов в минуту с одного IP


    Args:
        req (CheckPlagiarismRequest):
            - user_text (str): Текст пользователя
            - get_index (bool): Если True: ответ будет 0 или 1, Если False: AI, HUMAN

    Returns:
        Возвращает AI or HUMAN

    Raises:
        Пока отсутствуют
    """
    label = anti_plag.get_label(req.user_text)

    if req.get_index:
        label = anti_plag.get_index_from_label(label)

    return CheckPlagiarismResponse(label=label)