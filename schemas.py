"""
schemas.py — Pydantic request/response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Request body for a single string comparison."""

    string_a: str = Field(
        ...,
        min_length=1,
        description="First string to compare.",
        examples=["Abhinav K Baliyan"],
    )
    string_b: str = Field(
        ...,
        min_length=1,
        description="Second string to compare.",
        examples=["Abhinav Kumar Baliyan"],
    )
    weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional custom metric weights. Keys: levenshtein, jaro_winkler, "
            "jaccard, token_sort, token_set, embedding. "
            "Values are relative — they'll be auto-normalized to sum to 1.0."
        ),
        examples=[{"token_sort": 0.30, "embedding": 0.25, "levenshtein": 0.20}],
    )


class CompareResult(BaseModel):
    """Response body for a single comparison."""

    string_a: str = Field(..., description="First input string (echoed back).")
    string_b: str = Field(..., description="Second input string (echoed back).")
    score: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Final weighted similarity score (0–100).",
    )
    match_level: str = Field(
        default="no_match",
        description="Classification: exact_match, probable_match, possible_match, or no_match.",
    )
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Individual metric scores (0–100 each).",
    )
    string_a_normalized: str | None = Field(
        default=None,
        description="First string normalized by Bedrock."
    )
    string_b_normalized: str | None = Field(
        default=None,
        description="Second string normalized by Bedrock."
    )


class BatchCompareRequest(BaseModel):
    """Request body for batch comparison."""

    pairs: list[CompareRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of string pairs to compare. Maximum 1000 pairs per request.",
    )


class BatchCompareResult(BaseModel):
    """Response body for batch comparison."""

    results: list[CompareResult]
    count: int = Field(..., description="Number of pairs compared.")


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str = Field(..., description="Service status: 'healthy'.")
    fasttext_model_loaded: bool = Field(
        ...,
        description="Whether the FastText embedding model is loaded in memory.",
    )
    available_metrics: list[str] = Field(
        ...,
        description="List of similarity metrics available.",
    )
