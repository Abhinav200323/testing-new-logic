"""
embeddings.py — FastText word embeddings for semantic similarity.

Uses the pre-trained cc.en.300.bin model (300-dimensional word vectors
trained on Common Crawl). The model is loaded lazily on first use.

Key advantage: FastText uses subword (character n-gram) information,
so it can generate meaningful vectors even for misspelled words
and out-of-vocabulary tokens — critical for name/address matching.
"""

import os
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_ft_model = None
_model_load_attempted = False

# Default search paths for the FastText model
MODEL_FILENAME = "cc.en.300.bin"
MODEL_SEARCH_PATHS = [
    Path(__file__).parent.parent / MODEL_FILENAME,   # project root
    Path.home() / MODEL_FILENAME,                     # home directory
    Path.home() / ".fasttext" / MODEL_FILENAME,       # ~/.fasttext/
]


def _find_model_path() -> str | None:
    """Search known locations for the FastText model file."""
    # Check environment variable override first
    env_path = os.environ.get("FASTTEXT_MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    for path in MODEL_SEARCH_PATHS:
        if path.is_file():
            return str(path)

    return None


def get_model():
    """
    Lazily load the FastText model (singleton).

    Returns the model if available, None otherwise.
    Logs a warning on first failed attempt — does not crash the server.
    """
    global _ft_model, _model_load_attempted

    if _ft_model is not None:
        return _ft_model

    if _model_load_attempted:
        return None

    _model_load_attempted = True

    model_path = _find_model_path()
    if model_path is None:
        logger.warning(
            "FastText model '%s' not found. Embedding similarity will return 0.0. "
            "Download it with: python -c \"import fasttext.util; fasttext.util.download_model('en')\" "
            "or set FASTTEXT_MODEL_PATH environment variable.",
            MODEL_FILENAME,
        )
        return None

    try:
        import fasttext

        logger.info("Loading FastText model from %s (this may take a minute)...", model_path)
        _ft_model = fasttext.load_model(model_path)
        logger.info("FastText model loaded successfully — %d dimensions.", _ft_model.get_dimension())
        return _ft_model
    except Exception as e:
        logger.error("Failed to load FastText model: %s", e)
        return None


def is_model_loaded() -> bool:
    """Check if the FastText model is currently loaded in memory."""
    return _ft_model is not None


# ---------------------------------------------------------------------------
# Vector computation
# ---------------------------------------------------------------------------

def get_sentence_vector(tokens: list[str]) -> np.ndarray | None:
    """
    Compute a sentence vector by averaging individual token vectors.

    Args:
        tokens: List of normalized tokens.

    Returns:
        Averaged 300-d numpy vector, or None if model unavailable / empty tokens.
    """
    model = get_model()
    if model is None or not tokens:
        return None

    vectors = [model.get_word_vector(token) for token in tokens]
    return np.mean(vectors, axis=0)


def embedding_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Cosine similarity between averaged token embeddings (0–100 scale).

    Args:
        tokens_a: Tokens from string A.
        tokens_b: Tokens from string B.

    Returns:
        Cosine similarity * 100. Returns 0.0 if model is unavailable.
    """
    vec_a = get_sentence_vector(tokens_a)
    vec_b = get_sentence_vector(tokens_b)

    if vec_a is None or vec_b is None:
        return 0.0

    # Cosine similarity
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_sim = dot / (norm_a * norm_b)

    # Clamp to [0, 1] (cosine can be slightly negative for unrelated words)
    cosine_sim = max(0.0, min(1.0, float(cosine_sim)))

    return cosine_sim * 100
