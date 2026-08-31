from __future__ import annotations

from typing import Any


def gemma_auto_classes() -> tuple[Any, Any, Any]:
    """Return the current Gemma 4 multimodal loader classes."""
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install a Transformers release with Gemma 4 multimodal support.") from exc
    return AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
