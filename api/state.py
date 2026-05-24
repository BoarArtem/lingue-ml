# api/state.py
import torch
import joblib
from dataclasses import dataclass
from gensim.models import Word2Vec

from api.logger import build_logger
from models.b2_predictor import B2PredictorModel
from inference.omnivoice_tts_inference import OmniVoiceInference

logger = build_logger("ml_linguo")


@dataclass
class AppState:
    device: torch.device = None
    word2vec: Word2Vec | None = None
    b2: B2PredictorModel | None = None
    tts: OmniVoiceInference | None = None


def load_models(model_dir: str, device: torch.device) -> AppState:
    """
    Загружает все ML модели из указанной директории.
    Если одна модель упала — остальные продолжают грузиться.

    Args:
        model_dir: Путь к директории с весами моделей
        device: torch.device — cuda или cpu

    Returns:
        AppState с загруженными моделями (None если загрузка упала)
    """
    state = AppState(device=device)

    try:
        state.word2vec = Word2Vec.load(f"{model_dir}/word2vec.model")
        logger.info("Word2Vec loaded")
    except Exception:
        logger.error("Word2Vec load failed", exc_info=True)

    try:
        state.b2 = joblib.load(f"{model_dir}/b2_model.pkl")
        logger.info("B2Model loaded")
    except Exception:
        logger.error("B2Model load failed", exc_info=True)

    try:
        state.tts = OmniVoiceInference(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
        logger.info("OmniVoiceTTS loaded")
    except Exception:
        logger.error("OmniVoiceTTS load failed", exc_info=True)

    return state