"""Centralised logging configuration for MCNP Tools.

This module applies :func:`logging.basicConfig` with a default format and
level. The log level can be overridden via the ``LOG_LEVEL`` environment
variable or by passing an explicit level to :func:`configure`.
"""

from __future__ import annotations

import logging
import os
from typing import Union

DEFAULT_FORMAT = "%(levelname)s:%(name)s:%(message)s"


def configure(level: Union[str, int, None] = None) -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Logging level to apply. If ``None``, the ``LOG_LEVEL`` environment
        variable is consulted and defaults to ``INFO`` if unset.
    """

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=DEFAULT_FORMAT, force=True)
