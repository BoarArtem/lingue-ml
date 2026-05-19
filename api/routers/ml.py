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
    label = anti_plag.get_label(req.user_text)

    if req.get_index:
        label = anti_plag.get_index_from_label(label)

    return CheckPlagiarismResponse(label=label)