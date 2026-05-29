"""
similarity.py — Character-level, token-level, and phonetic similarity metrics.

Metrics:
- Levenshtein ratio (RapidFuzz)
- Jaro-Winkler similarity (Jellyfish)
- Jaccard similarity (pure Python, token-level)
- Token Sort Ratio (RapidFuzz)
- Token Set Ratio (RapidFuzz)
- Soundex match score (Jellyfish)
"""

import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from engine.normalizer import normalize, tokenize


# ---------------------------------------------------------------------------
# Character-level metrics
# ---------------------------------------------------------------------------

def levenshtein_similarity(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity (0–100 scale).

    Uses RapidFuzz's optimized C++ implementation.
    Good for catching typos and minor character edits.
    """
    norm_a = normalize(a)
    norm_b = normalize(b)
    if not norm_a and not norm_b:
        return 100.0
    if not norm_a or not norm_b:
        return 0.0
    return Levenshtein.normalized_similarity(norm_a, norm_b) * 100


def jaro_winkler_similarity(a: str, b: str) -> float:
    """
    Jaro-Winkler similarity (0–100 scale).

    Prefix-weighted — gives bonus to strings sharing a common prefix.
    Especially effective for person names (first name matching).
    """
    norm_a = normalize(a)
    norm_b = normalize(b)
    if not norm_a and not norm_b:
        return 100.0
    if not norm_a or not norm_b:
        return 0.0
    return jellyfish.jaro_winkler_similarity(norm_a, norm_b) * 100


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------

def jaccard_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity on token sets (0–100 scale).

    |intersection| / |union| of the token sets.
    Handles reordered tokens and ignores duplicates.
    """
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a and not tokens_b:
        return 100.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return (len(intersection) / len(union)) * 100


def token_sort_similarity(a: str, b: str) -> float:
    """
    Token Sort Ratio (0–100 scale).

    Sorts tokens alphabetically before comparing.
    Handles reordered words: "Kumar Abhinav" vs "Abhinav Kumar" → ~100.
    """
    norm_a = normalize(a)
    norm_b = normalize(b)
    if not norm_a and not norm_b:
        return 100.0
    if not norm_a or not norm_b:
        return 0.0
    return fuzz.token_sort_ratio(norm_a, norm_b)


def token_set_similarity(a: str, b: str) -> float:
    """
    Token Set Ratio (0–100 scale).

    Compares the intersection and remainder of token sets.
    Handles extra tokens: "Abhinav Kumar Baliyan" vs "Abhinav Baliyan" → high score.
    """
    norm_a = normalize(a)
    norm_b = normalize(b)
    if not norm_a and not norm_b:
        return 100.0
    if not norm_a or not norm_b:
        return 0.0
    return fuzz.token_set_ratio(norm_a, norm_b)


# ---------------------------------------------------------------------------
# Phonetic metrics
# ---------------------------------------------------------------------------

def soundex_similarity(a: str, b: str) -> float:
    """
    Soundex-based phonetic similarity (0–100 scale).

    Compares Soundex codes of individual tokens using best-match pairing.
    Effective for phonetically similar names: "Jon" vs "John" → high score.

    Strategy: For each token in A, find the best Soundex match in B.
    Average the best-match scores.
    """
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)

    if not tokens_a and not tokens_b:
        return 100.0
    if not tokens_a or not tokens_b:
        return 0.0

    def _soundex_token_score(t1: str, t2: str) -> float:
        """Compare two individual tokens by Soundex code."""
        try:
            s1 = jellyfish.soundex(t1)
            s2 = jellyfish.soundex(t2)
            if s1 == s2:
                return 100.0
            # Partial credit: same first letter
            if s1[0] == s2[0]:
                return 50.0
            return 0.0
        except Exception:
            return 0.0

    # For each token in A, find best match in B
    scores = []
    for ta in tokens_a:
        best = max(_soundex_token_score(ta, tb) for tb in tokens_b)
        scores.append(best)

    # For each token in B, find best match in A (symmetric)
    for tb in tokens_b:
        best = max(_soundex_token_score(tb, ta) for ta in tokens_a)
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0.0
