"""
Structured logging configuration using loguru.
Outputs JSON in production, human-readable in development.
"""
import sys
from loguru import logger
from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()

    if settings.debug:
        # Development: coloured, human-readable
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "{message}"
            ),
            colorize=True,
        )
    else:
        # Production: JSON structured logs
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format="{time} | {level} | {name}:{function}:{line} | {message}",
            serialize=True,
        )

    logger.info(
        "Logging configured",
        level=settings.log_level,
        debug=settings.debug,
    )


__all__ = ["logger", "setup_logging"]
