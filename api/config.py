from pydantic_settings import BaseSettings # добавить

import os

class Settings(BaseSettings):
    model_dir: str = os.getenv("MODEL_DIR", "storage/models")

    # app version
    APP_VERSION: str = "v2.11.5"

    # spam-model
    spam_vocab_size: int = 10000
    spam_embed_dim: int = 128
    spam_hidden_size: int = 256
    spam_num_layers: int = 2

    # rate limits
    rate_limit_similar: str = "30/minute"
    rate_limit_llm: str = "10/minute"

    # logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
