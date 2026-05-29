"""
normalizer.py — Input normalization and tokenization.

Handles:
- Unicode normalization (NFKD)
- Lowercasing
- Punctuation removal
- Whitespace collapsing
- Token splitting
"""

import re
import unicodedata


def normalize(text: str) -> str:
    """
    Normalize a string for similarity comparison.

    Steps:
        1. Unicode NFKD normalization (decomposes ligatures, accents, etc.)
        2. Strip combining characters (accents like é → e)
        3. Lowercase
        4. Remove all punctuation (keep alphanumeric + spaces)
        5. Collapse multiple spaces into one
        6. Strip leading/trailing whitespace

    Args:
        text: Raw input string.

    Returns:
        Cleaned, normalized string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Unicode NFKD normalization
    text = unicodedata.normalize("NFKD", text)

    # Strip combining characters (accents)
    text = "".join(
        ch for ch in text if not unicodedata.combining(ch)
    )

    # Lowercase
    text = text.lower()

    # Remove punctuation — keep only alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenize a normalized string into individual tokens.

    Args:
        text: A normalized string (should be run through normalize() first).

    Returns:
        List of non-empty string tokens.
    """
    normalized = normalize(text)
    if not normalized:
        return []
    return normalized.split()
