import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.config import settings
from api.state import load_models
from api.logger import build_logger
from api.routers import nlp, ml, llm, system

APP_VERSION = "v2.11.5"

logger = build_logger("ml_linguo")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ml = load_models(settings.model_dir, device)
    logger.info("Service started", extra={"version": APP_VERSION})
    yield
    logger.info("Service shutting down")


app = FastAPI(
    title="ML Linguo Service",
    version=APP_VERSION,
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(system.router)
app.include_router(nlp.router)
app.include_router(ml.router)
app.include_router(llm.router)