"""
scorer.py — Feature extraction and weighted scoring.

Orchestrates all similarity metrics and produces a final score
with match-level classification.
"""

from engine.normalizer import tokenize
from engine.similarity import (
    levenshtein_similarity,
    jaro_winkler_similarity,
    jaccard_similarity,
    token_sort_similarity,
    token_set_similarity,
    soundex_similarity,
)
from engine.embeddings import embedding_similarity, is_model_loaded


# ---------------------------------------------------------------------------
# Default weights (tuned for name/address matching)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "token_sort": 0.25,
    "embedding": 0.20,
    "levenshtein": 0.20,
    "token_set": 0.15,
    "jaccard": 0.10,
    "jaro_winkler": 0.10,
}

# Metrics that should be excluded from scoring when their backing model isn't loaded
_MODEL_DEPENDENT_METRICS = {"embedding"}

# Match level thresholds
THRESHOLDS = {
    "exact_match": 95.0,
    "probable_match": 80.0,
    "possible_match": 70.0,
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(string_a: str, string_b: str) -> dict[str, float]:
    """
    Run all similarity metrics on a pair of strings.

    Args:
        string_a: First string.
        string_b: Second string.

    Returns:
        Dictionary of metric_name → score (0–100 scale).
    """
    tokens_a = tokenize(string_a)
    tokens_b = tokenize(string_b)

    features = {
        "levenshtein": levenshtein_similarity(string_a, string_b),
        "jaro_winkler": jaro_winkler_similarity(string_a, string_b),
        "jaccard": jaccard_similarity(string_a, string_b),
        "token_sort": token_sort_similarity(string_a, string_b),
        "token_set": token_set_similarity(string_a, string_b),
        "soundex": soundex_similarity(string_a, string_b),
        "embedding": embedding_similarity(tokens_a, tokens_b),
    }

    # Round all scores to 2 decimal places
    return {k: round(v, 2) for k, v in features.items()}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(
    features: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute the weighted aggregate similarity score.

    If the FastText model is not loaded, the embedding metric is automatically
    excluded and its weight is redistributed among the remaining metrics.

    Args:
        features: Dictionary from extract_features().
        weights: Optional custom weight overrides. Keys must match feature names.
                 Missing keys fall back to DEFAULT_WEIGHTS.
                 Weights are auto-normalized to sum to 1.0.

    Returns:
        Weighted similarity score (0–100 scale).
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Exclude model-dependent metrics if the model isn't available
    if not is_model_loaded():
        for metric in _MODEL_DEPENDENT_METRICS:
            w.pop(metric, None)

    # Normalize weights to sum to 1.0
    total_weight = sum(w.values())
    if total_weight == 0:
        return 0.0

    score = 0.0
    for metric, weight in w.items():
        if metric in features:
            score += (weight / total_weight) * features[metric]

    return round(score, 2)


def classify_match(score: float) -> str:
    """
    Classify a score into a match level.

    Thresholds:
        >= 95  → exact_match
        85–94  → probable_match
        70–84  → possible_match
        < 70   → no_match

    Args:
        score: Similarity score (0–100).

    Returns:
        Match level string.
    """
    if score >= THRESHOLDS["exact_match"]:
        return "exact_match"
    elif score >= THRESHOLDS["probable_match"]:
        return "probable_match"
    elif score >= THRESHOLDS["possible_match"]:
        return "possible_match"
    else:
        return "no_match"


def compare(
    string_a: str,
    string_b: str,
    weights: dict[str, float] | None = None,
) -> dict | str:
    """
    Full comparison pipeline: extract features → compute score → classify.

    If custom weights are not provided, it auto-detects Address vs Name logic based
    on word count. If 4 or more words, applies address-heavy weights.
    
    Now includes a Bedrock preprocessing step that expands abbreviations and checks
    if the strings are fundamentally similar before running the metrics.

    Args:
        string_a: First string.
        string_b: Second string.
        weights: Optional custom weights.

    Returns:
        Dictionary with score, match_level, and per-metric features, OR
        a string 'string dont match ' if Bedrock determines they are entirely different.
    """
    from engine.bedrock_preprocessor import preprocess_strings

    # 1. Preprocess with Bedrock Converse API
    pre_result = preprocess_strings(string_a, string_b)
    
    if not pre_result.get("is_similar", True):
        return "string dont match "
        
    # Use the normalized strings for metrics calculation
    norm_a = pre_result.get("string_a_normalized", string_a)
    norm_b = pre_result.get("string_b_normalized", string_b)

    if weights is None:
        tokens_a = tokenize(norm_a)
        tokens_b = tokenize(norm_b)
        max_words = max(len(tokens_a), len(tokens_b))
        
        # If 4 or more words, assume it's an address or heavy entity
        if max_words >= 4:
            weights = {
                "token_set": 0.40,
                "token_sort": 0.35,
                "embedding": 0.10,
                "jaccard": 0.10,
                "jaro_winkler": 0.05,
                "levenshtein": 0.00,
            }

    features = extract_features(norm_a, norm_b)
    score = compute_score(features, weights)
    match_level = classify_match(score)

    return {
        "score": score,
        "match_level": match_level,
        "features": features,
        "string_a_normalized": norm_a,
        "string_b_normalized": norm_b,
    }
