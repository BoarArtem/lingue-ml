import logging
import logging.handlers
import os
import json
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Форматирует каждую запись лога как JSON-строку.
    Удобно парсить в Datadog, Loki, ELK и т.д.
    """

    EXTRA_FIELDS = (
        "request_id", "method", "path", "status",
        "duration_ms", "client_ip", "word", "language",
        "topic", "prediction", "version",
    )

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "src_module": record.module,
            "src_func": record.funcName,
            "src_line": record.lineno,
        }

        for key in self.EXTRA_FIELDS:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def build_logger(name: str = "ml_linguo") -> logging.Logger:
    """
    Строит logger с тремя хендлерами:
    - StreamHandler (консоль) — читаемый формат для разработки
    - RotatingFileHandler (logs/app.log) — JSON, все уровни, ротация 10MB × 5
    - RotatingFileHandler (logs/errors.log) — JSON, только WARNING+, ротация 5MB × 3

    Args:
        name: Имя логгера, по умолчанию "ml_linguo"

    Returns:
        Настроенный logging.Logger
    """
    os.makedirs("logs", exist_ok=True)

    log = logging.getLogger(name)

    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)
    log.propagate = False

    _add_stream_handler(log)
    _add_file_handler(log, path="logs/app.log", level=logging.DEBUG,
                      max_bytes=10 * 1024 * 1024, backup_count=5)
    _add_file_handler(log, path="logs/errors.log", level=logging.WARNING,
                      max_bytes=5 * 1024 * 1024, backup_count=3)

    return log


def _add_stream_handler(log: logging.Logger) -> None:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    log.addHandler(handler)


def _add_file_handler(
        log: logging.Logger,
        path: str,
        level: int,
        max_bytes: int,
        backup_count: int,
) -> None:
    handler = logging.handlers.RotatingFileHandler(
        filename=path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    log.addHandler(handler)