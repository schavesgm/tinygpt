"""Module containing functionality to log information from the experiments."""

from __future__ import annotations

import logging
import sys

__all__ = ["get_logger"]

LOGGING_FORMAT: logging.Formatter = logging.Formatter(
    fmt="%(levelname)s - [%(asctime)s — %(funcName)s:%(lineno)d]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a new ``Logger`` object with the provided name.

    Args:
        name (str): Logger instance name.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name=name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    # Console logging handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(LOGGING_FORMAT)
    logger.addHandler(console_handler)

    return logger
