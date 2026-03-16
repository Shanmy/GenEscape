"""
Configuration module for GenEscape.
All settings are read from environment variables with sensible defaults.
"""

import os
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


# API Keys
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")

# Provider selection
_provider_str: str = os.environ.get("LLM_PROVIDER", "openai").lower()
DEFAULT_PROVIDER: LLMProvider = (
    LLMProvider.GEMINI if _provider_str == "gemini" else LLMProvider.OPENAI
)

# Model names
OPENAI_TEXT_MODEL: str = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o")
OPENAI_IMAGE_MODEL: str = os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3")
GEMINI_TEXT_MODEL: str = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_IMAGE_MODEL: str = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

# Pipeline settings
MAX_REFINEMENT_ITERS: int = int(os.environ.get("MAX_REFINEMENT_ITERS", "5"))
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "output")
