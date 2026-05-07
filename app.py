import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from gensim.models import Word2Vec
import os
import nltk
import uuid
import traceback

from inference.topic_predictor import TopicPredictor
from models import anti_plagiarism
from models.b2_predictor import B2PredictorModel
from models.llm_sentence_generate import llm_sentence_generate
from models.llm_word_level import llm_word_level
from models.llm_correct_paragraph import correct_paragraph, get_changed_word, word_pair
from inference.anti_plagiarism_model_inference import AntiPlagiarismModelInference

from data.tokenizer import (
    sentence_preprocess_english,
    sentence_preprocess_russian,
    sentence_preprocess_spanish,
    sentence_preprocess_france,
    sentence_preprocess_german,
    sentence_preprocess_chinese
)

import logging
import logging.handlers
import json
import time
from datetime import datetime, timezone

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


class JSONFormatter(logging.Formatter):
    """
    Форматирует каждую запись лога как JSON-строку.
    Это позволяет легко парсить логи в Datadog, Loki, ELK и т.д.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts":         datetime.now(timezone.utc).isoformat(),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.getMessage(),
            "src_module": record.module,
            "src_func":   record.funcName,
            "src_line":   record.lineno,
        }

        for key in ("request_id", "method", "path", "status", "duration_ms",
                    "client_ip", "word", "language", "topic", "prediction"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def build_logger(name: str = "ml_linguo") -> logging.Logger:
    """
    Строит logger с тремя хендлерами:
    - StreamHandler (консоль) — читаемый формат для разработки
    - RotatingFileHandler (файл) — JSON, ротация 10MB × 5 файлов
    - Отдельный файл для ошибок (только WARNING+)
    """
    os.makedirs("logs", exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    file_handler = logging.handlers.RotatingFileHandler(
        filename="logs/app.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    error_handler = logging.handlers.RotatingFileHandler(
        filename="logs/errors.log",
        maxBytes=5 * 1024 * 1024,    # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(JSONFormatter())

    log.addHandler(stream_handler)
    log.addHandler(file_handler)
    log.addHandler(error_handler)

    return log


logger = build_logger("ml_linguo")


APP_VERSION = "v2.11.5"

app = FastAPI(
    title="ML Linguo Service",
    description="""
ML сервис для Linguo.

Возможности API:

• поиск похожих слов (FastText embeddings)  
• определение уровня слова (CEFR)  
• генерация предложений  
• ML предсказания  
• preprocessing текста  
""",
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://34.30.102.15",
        "http://34.10.240.6",
        "https://api.ml.linguo.foo",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


model_dir = os.getenv("MODEL_DIR", "/models")

logger.info("Loading Word2Vec model...")
ve_model = Word2Vec.load(f"{model_dir}/word2vec.model")

try:
    topic_predictor = TopicPredictor()
    logger.info("TopicPredictor loaded successfully")
except Exception as e:
    logger.error("Failed to load TopicPredictor", exc_info=True)
    topic_predictor = None

try:
    predictor: B2PredictorModel = joblib.load(f"{model_dir}/b2_model.pkl")
    logger.info("B2PredictorModel loaded from disk")
except FileNotFoundError:
    predictor = B2PredictorModel()
    logger.warning("B2PredictorModel not found on disk — using untrained instance")

try:
    anti_plagiarism = AntiPlagiarismModelInference()
    logger.info("AntiPlagiarismModelInference loaded successfully")
except Exception as e:
    logger.error("Failed to load AntiPlagiarismModelInference", exc_info=True)
    anti_plagiarism = None

class TopicRequest(BaseModel):
    sentences: list[str] = Field(..., example=["I love coding in Python"])

class SingleTopicRequest(BaseModel):
    sentence: str = Field(..., example="I love coding in Python")

class PredictRequest(BaseModel):
    features: dict = Field(..., example={
        'unique_words': 1500, 'words_a1': 600, 'words_a2': 500,
        'words_b1': 400, 'words_b2': 0, 'avg_acc_7d': 0.88,
        'avg_acc_30d': 0.85, 'avg_time_sec': 6.0,
        'words_day_7d': 30, 'words_day_30d': 900,
        'streak': 20, 'sessions_week': 14
    })

class SimilarRequest(BaseModel):
    arr: list[str] = Field(..., example=["dog", "cat"])
    topn: int = Field(default=10, example=5)

class WordLevelRequest(BaseModel):
    word: str = Field(example="nevertheless")
    translation: str = Field(example="тем не менее")

class SentenceRequest(BaseModel):
    word: str = Field(example="dog")
    level: str = Field(example="A1")
    language: str = Field(example="en | English")

class PreprocessRequest(BaseModel):
    sentence: str = Field(example="Dogs are running in the park")
    language: str = Field(example="en")

class CorrectParagraphRequest(BaseModel):
    user_sentence: str = Field(example="I ate pizza yesterday")

class SentenceLevelRequest(BaseModel):
    user_sentence: str = Field(example="I ate apple yesterday")

class SentenceContextLevel(BaseModel):
    user_sentence: str = Field(example="I ate apple yesterday")

class SentenceContextRate(BaseModel):
    word: str = Field(example="go home")
    user_sentence: str = Field(example="I will go home tomorrow")

class CheckPlagiarismRequest(BaseModel):
    user_text: str = Field(example="bla bla bla bla bla bla bla bla")
    get_index: bool = Field(default=False)


@app.on_event("startup")
async def startup_event():
    logger.info(
        "Service started",
        extra={
            "version":        APP_VERSION,
            "model_dir":      model_dir,
            "word2vec_ok":    ve_model is not None,
            "b2_ok":          predictor.model is not None,
            "topic_ok":       topic_predictor is not None,
        }
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Service shutting down gracefully")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]          # короткий ID для трассировки
    request.state.request_id = request_id

    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    method = request.method

    logger.info(
        f"→ {method} {path}",
        extra={
            "request_id": request_id,
            "method":     method,
            "path":       path,
            "client_ip":  client_ip,
        }
    )

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.error(
            f"✗ {method} {path} — unhandled exception",
            exc_info=True,
            extra={
                "request_id":  request_id,
                "method":      method,
                "path":        path,
                "duration_ms": duration_ms,
                "client_ip":   client_ip,
            }
        )
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    status = response.status_code
    level = logging.WARNING if status >= 400 else logging.INFO

    logger.log(
        level,
        f"← {method} {path} {status} ({duration_ms}ms)",
        extra={
            "request_id":  request_id,
            "method":      method,
            "path":        path,
            "status":      status,
            "duration_ms": duration_ms,
            "client_ip":   client_ip,
        }
    )

    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", tags=["System"])
def health(request: Request):
    status = {
        "status":      "ok",
        "version":     APP_VERSION,
        "word2vec":    ve_model is not None,
        "b2_model":    predictor.model is not None,
        "topic_model": topic_predictor is not None,
    }
    logger.debug("Health check", extra={"request_id": getattr(request.state, "request_id", "-")})
    return status


@app.post("/similar", tags=["Embeddings"],
          summary="Поиск похожих слов")
@limiter.limit("30/minute")
def similar(request: Request, req: SimilarRequest):
    rid = getattr(request.state, "request_id", "-")

    if ve_model is None:
        logger.error("Word2Vec model not loaded", extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Word2Vec model not loaded")

    logger.info(
        f"Similar words lookup: words={req.arr}, topn={req.topn}",
        extra={"request_id": rid}
    )
    t0 = time.perf_counter()
    try:
        result = ve_model.wv.most_similar(req.arr, topn=req.topn)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            f"Similar words found: count={len(result)}, took={ms}ms",
            extra={"request_id": rid, "duration_ms": ms}
        )
        return result
    except KeyError as e:
        logger.warning(f"Word not in vocabulary: {e}", extra={"request_id": rid})
        raise HTTPException(status_code=404, detail=f"Word not found: {e}")
    except Exception:
        logger.error("Similar words error", exc_info=True, extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Internal error")


@app.post("/word_level", tags=["LLM"],
          summary="Определить уровень слова CEFR")
def word_level(request: Request, req: WordLevelRequest):
    rid = getattr(request.state, "request_id", "-")
    logger.info(
        f"Word level request: word='{req.word}'",
        extra={"request_id": rid, "word": req.word}
    )
    t0 = time.perf_counter()
    try:
        result = llm_word_level(req.word, req.translation)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            f"Word level result: word='{req.word}' → {result}, took={ms}ms",
            extra={"request_id": rid, "word": req.word, "duration_ms": ms}
        )
        return result
    except Exception:
        logger.error(f"Word level error for '{req.word}'", exc_info=True,
                     extra={"request_id": rid, "word": req.word})
        raise HTTPException(status_code=500, detail="LLM error")


@app.post("/sentence", tags=["LLM"],
          summary="Сгенерировать предложение")
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
        ms = round((time.perf_counter() - t0) * 1000, 1)
        preview = result[:60] + ("…" if len(result) > 60 else "")
        logger.info(
            f"Sentence generated: '{preview}', took={ms}ms",
            extra={"request_id": rid, "duration_ms": ms}
        )
        return result
    except Exception:
        logger.error("Sentence generation error", exc_info=True,
                     extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="LLM error")


@app.post("/predict", tags=["Machine Learning"],
          summary="ML предсказание B2")
def predict(request: Request, req: PredictRequest):
    rid = getattr(request.state, "request_id", "-")

    if not predictor.feature_names:
        logger.error("B2 model is not trained", extra={"request_id": rid})
        raise HTTPException(status_code=400, detail="Модель не обучена")

    logger.info(
        f"B2 prediction: features={list(req.features.keys())}",
        extra={"request_id": rid}
    )
    t0 = time.perf_counter()
    try:
        df = pd.DataFrame([req.features])
        missing_cols = [c for c in predictor.feature_names if c not in df.columns]
        if missing_cols:
            logger.warning(
                f"Missing columns: {missing_cols}",
                extra={"request_id": rid}
            )
            raise HTTPException(status_code=400,
                                detail=f"Отсутствуют колонки: {missing_cols}")

        df = df[predictor.feature_names]
        pred = predictor.model.predict(df)[0]
        ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            f"B2 prediction result: {int(pred)}, took={ms}ms",
            extra={"request_id": rid, "prediction": int(pred), "duration_ms": ms}
        )
        return {"prediction": int(pred)}
    except HTTPException:
        raise
    except Exception:
        logger.error("B2 prediction error", exc_info=True,
                     extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Prediction error")


@app.post("/preprocess", tags=["NLP"],
          summary="Предобработка предложения")
def preprocess(request: Request, req: PreprocessRequest):
    rid = getattr(request.state, "request_id", "-")
    preprocessors = {
        "en": sentence_preprocess_english,
        "ru": sentence_preprocess_russian,
        "es": sentence_preprocess_spanish,
        "fr": sentence_preprocess_france,
        "de": sentence_preprocess_german,
        "ch": sentence_preprocess_chinese
    }

    if req.language not in preprocessors:
        logger.warning(
            f"Unsupported language: '{req.language}'",
            extra={"request_id": rid, "language": req.language}
        )
        raise HTTPException(status_code=400,
                            detail=f"Unsupported language: {req.language}")

    logger.info(
        f"Preprocessing: lang={req.language}, chars={len(req.sentence)}",
        extra={"request_id": rid, "language": req.language}
    )
    t0 = time.perf_counter()
    try:
        result = preprocessors[req.language](req.sentence)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            f"Preprocessed: tokens={len(result)}, took={ms}ms",
            extra={"request_id": rid, "duration_ms": ms}
        )
        return result
    except Exception:
        logger.error("Preprocessing error", exc_info=True,
                     extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Preprocessing error")


@app.post("/predict_topic", tags=["Machine Learning"],
          summary="Определение темы — одно предложение")
@limiter.limit("10/minute")
def predict_topic(request: Request, req: SingleTopicRequest):
    rid = getattr(request.state, "request_id", "-")

    if not topic_predictor:
        logger.error("TopicPredictor not initialized", extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Topic model is not initialized")

    logger.info(
        f"Topic prediction: chars={len(req.sentence)}",
        extra={"request_id": rid}
    )
    t0 = time.perf_counter()
    try:
        result = topic_predictor.get_topic(req.sentence)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            f"Topic predicted: '{result}', took={ms}ms",
            extra={"request_id": rid, "topic": result, "duration_ms": ms}
        )
        return {"topic": result}
    except Exception:
        logger.error("Topic prediction error", exc_info=True,
                     extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Topic prediction error")


@app.post("/predict_topics", tags=["Machine Learning"],
          summary="Определение тем — массив предложений")
@limiter.limit("10/minute")
def predict_topics(request: Request, req: TopicRequest):
    rid = getattr(request.state, "request_id", "-")

    if not topic_predictor:
        logger.error("TopicPredictor not initialized", extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Topic model is not initialized")

    logger.info(
        f"Topics prediction: sentences={len(req.sentences)}",
        extra={"request_id": rid}
    )
    t0 = time.perf_counter()
    try:
        results = topic_predictor.get_topics(req.sentences)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            f"Topics predicted: count={len(results)}, took={ms}ms",
            extra={"request_id": rid, "duration_ms": ms}
        )
        return {"topics": results}
    except Exception:
        logger.error("Topics prediction error", exc_info=True,
                     extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Topics prediction error")


@app.post("/correct_paragraph", tags=["LLM"],
          summary="Исправление ошибок в предложении")
@limiter.limit("10/minute")
def correct_paragraph_checking(request: Request, req: CorrectParagraphRequest):
    rid = getattr(request.state, "request_id", "-")
    logger.info(
        f"Correct paragraph: chars={len(req.user_sentence)}",
        extra={"request_id": rid}
    )
    t0 = time.perf_counter()
    try:
        ai_sentence = correct_paragraph(req.user_sentence)
        incorrect_words, correct_words = get_changed_word(req.user_sentence, ai_sentence)
        ms = round((time.perf_counter() - t0) * 1000, 1)

        changes = len(incorrect_words)
        logger.info(
            f"Paragraph corrected: changes={changes}, took={ms}ms",
            extra={"request_id": rid, "duration_ms": ms}
        )
        return {
            "User sentence":  req.user_sentence,
            "AI sentence":    ai_sentence,
            "Changing pair":  word_pair(incorrect_words, correct_words)
        }
    except Exception:
        logger.error("Correct paragraph error", exc_info=True,
                     extra={"request_id": rid})
        raise HTTPException(status_code=500, detail="Correction error")


@app.post("/sentence_level", tags=["LLM"],
          summary="Уровень предложения")
def sentence_level(request: Request, req: SentenceLevelRequest):
    logger.warning(
        "sentence_level called but not implemented",
        extra={"request_id": getattr(request.state, "request_id", "-")}
    )
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/sentence_context_level", tags=["LLM"],
          summary="Уровень предложения по слову в контексте")
def sentence_context_level(request: Request, req: SentenceContextRate):
    logger.warning(
        "sentence_context_level called but not implemented",
        extra={"request_id": getattr(request.state, "request_id", "-")}
    )
    return {"status": "ok"}


@app.post("/sentence_context_rate", tags=["LLM"],
          summary="Оценка использования слова в контексте")
def sentence_context_rate(request: Request, req: SentenceContextRate):
    logger.warning(
        "sentence_context_rate called but not implemented",
        extra={"request_id": getattr(request.state, "request_id", "-")}
    )
    raise HTTPException(status_code=501, detail="Not implemented")

@app.post("/check_plagiarism", tags=["Machine Learning"],
          summary="Проверка текста на AI-плагиат")
@limiter.limit("10/minute")
def check_plagiarism(request: Request, req: CheckPlagiarismRequest):

    if req.user_text is None:
        logger.warning("user_text is required")
        raise HTTPException(status_code=400, detail="user_text is required")

    if anti_plagiarism is None:
        logger.error("Anti-plagiarism model is not initialized")
        raise HTTPException(status_code=500, detail="Anti-plagiarism model is not initialized")

    try:
        label = anti_plagiarism.get_label(req.user_text)
    except Exception as e:
        logger.error("Error in check_plagiarism", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in check_plagiarism: {e}")

    if req.get_index:
        try:
            label = anti_plagiarism.get_index_from_label(label)
        except Exception as e:
            logger.error("Error in get_index_from_label", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error in get_index_from_label: {e}")

    logger.info(f"Plagiarism check result: {label}")

    return {"label": label}
