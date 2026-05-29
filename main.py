"""
main.py — FastAPI application for the Hybrid String Similarity System.

Endpoints:
    POST /api/v1/compare       — Compare a single pair of strings
    POST /api/v1/compare/batch — Compare multiple pairs in one request
    GET  /api/v1/health        — Health check + model status
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    CompareRequest,
    CompareResult,
    BatchCompareRequest,
    BatchCompareResult,
    HealthResponse,
)
from engine.scorer import compare, DEFAULT_WEIGHTS
from engine.embeddings import is_model_loaded, get_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("similarity-api")

# ---------------------------------------------------------------------------
# Lifespan — optional model preloading
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Attempt to preload the FastText model on startup."""
    logger.info("Starting Hybrid String Similarity API...")
    logger.info("Attempting to preload FastText model...")
    model = get_model()
    if model:
        logger.info("FastText model preloaded successfully.")
    else:
        logger.warning(
            "FastText model not available — embedding similarity will return 0.0. "
            "Other metrics will work normally."
        )
    yield
    logger.info("Shutting down Hybrid String Similarity API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hybrid String Similarity API",
    description=(
        "A production-grade string similarity system using multiple metrics "
        "(Levenshtein, Jaro-Winkler, Jaccard, Token Sort/Set, Soundex, FastText embeddings). "
        "Designed for name and address matching without LLMs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Available metrics list (for health endpoint)
# ---------------------------------------------------------------------------

AVAILABLE_METRICS = [
    "levenshtein",
    "jaro_winkler",
    "jaccard",
    "token_sort",
    "token_set",
    "soundex",
    "embedding",
]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/compare",
    response_model=CompareResult,
    summary="Compare two strings",
    tags=["Similarity"],
)
async def compare_strings(request: CompareRequest) -> CompareResult:
    """
    Compare two strings and return a similarity score with per-metric breakdown.

    The final score is a weighted combination of 7 similarity metrics.
    Custom weights can be provided to adjust metric importance.
    """
    try:
        result = compare(
            string_a=request.string_a,
            string_b=request.string_b,
            weights=request.weights,
        )
        if isinstance(result, str):
            return CompareResult(
                string_a=request.string_a,
                string_b=request.string_b,
                match_level=result,
            )
        return CompareResult(
            string_a=request.string_a,
            string_b=request.string_b,
            **result,
        )
    except Exception as e:
        logger.error("Comparison failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post(
    "/api/v1/compare/batch",
    response_model=BatchCompareResult,
    summary="Compare multiple string pairs",
    tags=["Similarity"],
)
async def compare_batch(request: BatchCompareRequest) -> BatchCompareResult:
    """
    Compare multiple pairs of strings in a single request.

    Maximum 1000 pairs per request. Each pair is scored independently.
    """
    try:
        results = []
        for pair in request.pairs:
            result = compare(
                string_a=pair.string_a,
                string_b=pair.string_b,
                weights=pair.weights,
            )
            if isinstance(result, str):
                results.append(
                    CompareResult(
                        string_a=pair.string_a,
                        string_b=pair.string_b,
                        match_level=result,
                    )
                )
            else:
                results.append(
                    CompareResult(
                        string_a=pair.string_a,
                        string_b=pair.string_b,
                        **result,
                    )
                )
        return BatchCompareResult(results=results, count=len(results))
    except Exception as e:
        logger.error("Batch comparison failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch comparison failed: {str(e)}")


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    Check the health of the service and whether the FastText model is loaded.
    """
    return HealthResponse(
        status="healthy",
        fasttext_model_loaded=is_model_loaded(),
        available_metrics=AVAILABLE_METRICS,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
