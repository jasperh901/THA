"""Thalamus Adapter (THA) reference implementation."""

from .config import RunConfig, compose_config
from .model import THAModel
from .modules import ThalamusAdapter

__all__ = ["RunConfig", "THAModel", "ThalamusAdapter", "compose_config"]
__version__ = "0.1.0"
