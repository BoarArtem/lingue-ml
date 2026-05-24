# api/dependencies.py
import torch
from fastapi import Request, HTTPException

from api.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.ml


def get_device(request: Request) -> torch.device:
    return request.app.state.ml.device


def get_word2vec(request: Request):
    model = request.app.state.ml.word2vec
    if model is None:
        raise HTTPException(500, "Word2Vec not loaded")
    return model


def get_b2_model(request: Request):
    model = request.app.state.ml.b2
    if model is None:
        raise HTTPException(500, "B2 model not loaded")
    return model


def get_tts(request: Request):
    model = request.app.state.ml.tts
    if model is None:
        raise HTTPException(500, "TTS model not loaded")
    return model