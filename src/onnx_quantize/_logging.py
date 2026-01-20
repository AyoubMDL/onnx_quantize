"""Logging utilities for onnx_quantize."""

import logging
import sys


__all__ = ["set_log_level"]


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)


def _configure_logger() -> None:
    """Configure the onnx_quantize logger with default settings."""
    logger = logging.getLogger("onnx_quantize")
    logger.setLevel(logging.INFO)

    # Only add handler if none exists (so we don't duplicate)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter("%(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        # Don't propagate to root logger to avoid duplicate messages
        logger.propagate = False


def set_log_level(level: int) -> None:
    """Set the verbosity level for onnx_quantize loggers.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG, logging.WARNING)
    """
    logger = logging.getLogger("onnx_quantize")
    logger.setLevel(level)


# Configure the library logger on import
_configure_logger()
