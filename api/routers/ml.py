import pandas as pd
from fastapi import APIRouter, Depends, Request
from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..dependencies import get_b2_model, get_word2vec
from ..logger import build_logger

from ..schemas import (
    PredictRequest, PredictResponse,
    SimilarResponse, SimilarRequest, SimilarItem,
)

router = APIRouter(tags=["Machine Learning"])
limiter = Limiter(key_func=get_remote_address)
logger = build_logger("ml_linguo")


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

    df = pd.DataFrame([req.features])
    missing = [c for c in b2.feature_names if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Отсутствуют колонки: {missing}")

    pred = b2.model.predict(df[b2.feature_names])[0]
    return PredictResponse(prediction=int(pred))


