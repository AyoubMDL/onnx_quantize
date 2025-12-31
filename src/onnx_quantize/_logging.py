"""Logging utilities for onnx_quantize."""

import logging
import sys


__all__ = ["set_log_level"]


def _configure_logger() -> None:
    """Configure the onnx_quantize logger with default settings."""
    logger = logging.getLogger("onnx_quantize")
    logger.setLevel(logging.INFO)

    # Only add handler if none exists (so we don't duplicate)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
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
