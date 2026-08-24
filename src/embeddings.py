#!/usr/bin/env python3
"""Embedding utilities using fastembed (ONNX-based, lightweight).

Supports multiple embedding models:
- TEXT: BAAI/bge-small-en-v1.5 (384-dim) — general text embedding
- CODE: jinaai/jina-embeddings-v2-base-code (768-dim) — code-aware embedding

GPU acceleration is available via the `fastembed-gpu` package with
providers=["CUDAExecutionProvider"]. Install with: uv pip install .[gpu]

On ARM64 (e.g., Apple Silicon), fastembed-gpu is not available, so the
CPU-only fastembed package is used automatically.
"""

import os
import platform
from enum import Enum
from functools import lru_cache

from fastembed import TextEmbedding

IS_ARM64 = platform.machine() in ("arm64", "aarch64")


class EmbeddingModel(Enum):
    TEXT = "BAAI/bge-small-en-v1.5"
    CODE = "jinaai/jina-embeddings-v2-base-code"


DIMENSIONS = {
    EmbeddingModel.TEXT: 384,
    EmbeddingModel.CODE: 768,
}

DEFAULT_MODEL = EmbeddingModel.TEXT


def _get_providers() -> list[str] | None:
    """Return ONNX execution providers based on environment and architecture.

    GPU providers are only available on x86_64 with fastembed-gpu installed.
    On ARM64, always returns None (CPU-only).
    """
    if IS_ARM64:
        return None
    if os.environ.get("FASTEMBED_GPU", "").lower() in ("1", "true", "yes"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return None


@lru_cache(maxsize=4)
def get_embedding_model(model: EmbeddingModel = DEFAULT_MODEL) -> TextEmbedding:
    """Load and cache an embedding model.

    Set FASTEMBED_GPU=1 to enable CUDA acceleration.
    """
    providers = _get_providers()
    kwargs = {"model_name": model.value}
    if providers:
        kwargs["providers"] = providers
    return TextEmbedding(**kwargs)


def get_dimensions(model: EmbeddingModel = DEFAULT_MODEL) -> int:
    """Return the output dimensionality for a model."""
    return DIMENSIONS[model]


def embed_text(text: str, model: EmbeddingModel = DEFAULT_MODEL) -> list[float]:
    """Generate embedding for a single text string."""
    m = get_embedding_model(model)
    embeddings = list(m.embed([text]))
    return embeddings[0].tolist()


def embed_texts(texts: list[str], model: EmbeddingModel = DEFAULT_MODEL) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    m = get_embedding_model(model)
    embeddings = list(m.embed(texts))
    return [e.tolist() for e in embeddings]
