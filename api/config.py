from pydantic_settings import BaseSettings # добавить

class Settings(BaseSettings):
    model_dir: str = "/models"

    # app version
    APP_VERSION: str = "v2.11.5"

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
