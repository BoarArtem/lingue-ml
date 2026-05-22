# api/state.py
import torch
import joblib
from dataclasses import dataclass
from gensim.models import Word2Vec

import torch

from api.logger import build_logger
from models.ml.spam_classification_model import SpamClassificationModel
from models.ml.b2_predictor import B2PredictorModel
from inference.topic_predictor import TopicPredictor
from inference.anti_plagiarism_model_inference import AntiPlagiarismModelInference
from inference.omnivoice_tts_inference import OmniVoiceInference
from datasets.spam_dataset_executor import vocab as spam_vocab_data

logger = build_logger("ml_linguo")


@dataclass
class AppState:
    device: torch.device = None
    word2vec: Word2Vec | None = None
    spam: SpamClassificationModel | None = None
    spam_vocab: dict | None = None
    b2: B2PredictorModel | None = None
    topic: TopicPredictor | None = None
    anti_plagiarism: AntiPlagiarismModelInference | None = None
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
    from api.config import settings

    state = AppState(device=device)

    try:
        state.word2vec = Word2Vec.load(f"{model_dir}/word2vec.model")
        logger.info("Word2Vec loaded")
    except Exception:
        logger.error("Word2Vec load failed", exc_info=True)

    try:
        spam = SpamClassificationModel(
            vocab_size=settings.spam_vocab_size,
            embed_dim=settings.spam_embed_dim,
            hidden_size=settings.spam_hidden_size,
            num_layers=settings.spam_num_layers,
        ).to(device)
        spam.load_state_dict(torch.load(
            f"{model_dir}/spam_classification_model_60.pth",
            map_location=device
        ))
        spam.eval()
        state.spam = spam
        state.spam_vocab = spam_vocab_data
        logger.info("SpamModel loaded")
    except Exception:
        logger.error("SpamModel load failed", exc_info=True)

    try:
        state.b2 = joblib.load(f"{model_dir}/b2_model.pkl")
        logger.info("B2Model loaded")
    except Exception:
        logger.error("B2Model load failed", exc_info=True)

    try:
        state.topic = TopicPredictor()
        logger.info("TopicPredictor loaded")
    except Exception:
        logger.error("TopicPredictor load failed", exc_info=True)

    try:
        state.anti_plagiarism = AntiPlagiarismModelInference()
        logger.info("AntiPlagiarism loaded")
    except Exception:
        logger.error("AntiPlagiarism load failed", exc_info=True)

    try:
        state.tts = OmniVoiceInference(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
        logger.info("OmniVoiceTTS loaded")
    except Exception:
        logger.error("OmniVoiceTTS load failed", exc_info=True)

    return state