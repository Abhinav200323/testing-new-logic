# Hybrid String Similarity System (GSE-v1)

A production-grade, modular string similarity engine that compares names, addresses, and other text entities **without LLMs or SLMs**. It combines 7 mathematical/textual similarity metrics into a single weighted score with explainable per-metric breakdowns.

## How It Works

```
Input Strings
     │
     ▼
┌─────────────────┐
│  Normalization   │  ← lowercase, unicode NFKD, strip punctuation, collapse spaces
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization    │  ← split into word tokens
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Feature Extraction (7 metrics)          │
│                                                      │
│  Character-level:  Levenshtein, Jaro-Winkler         │
│  Token-level:      Jaccard, Token Sort, Token Set    │
│  Phonetic:         Soundex                           │
│  Semantic:         FastText Embedding Cosine Sim     │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Weighted Score  │  ← configurable per-metric weights
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Match Level     │  ← exact / probable / possible / no_match
└─────────────────┘
```

## Similarity Metrics

| # | Metric | Library | What It Catches | Default Weight |
|---|--------|---------|-----------------|----------------|
| 1 | **Levenshtein Ratio** | RapidFuzz | Typos, minor edits (`Jon` → `John`) | 0.20 |
| 2 | **Jaro-Winkler** | Jellyfish | Prefix similarity (good for first names) | 0.10 |
| 3 | **Jaccard (tokens)** | Pure Python | Token overlap regardless of order | 0.10 |
| 4 | **Token Sort Ratio** | RapidFuzz | Reordered words (`Kumar Abhinav` → `Abhinav Kumar`) | 0.25 |
| 5 | **Token Set Ratio** | RapidFuzz | Extra/missing tokens (`Abhinav Baliyan` ↔ `Abhinav K Baliyan`) | 0.15 |
| 6 | **Soundex** | Jellyfish | Phonetically similar names (`Jon` ↔ `John`) | — |
| 7 | **FastText Embedding** | fasttext-wheel | Semantic/contextual similarity via subword vectors | 0.20 |

> **Note:** Soundex is included in the feature breakdown but not in the default weighted score. You can include it by passing custom weights.

### Scoring Formula

```
score = 0.25 × token_sort
      + 0.20 × embedding
      + 0.20 × levenshtein
      + 0.15 × token_set
      + 0.10 × jaccard
      + 0.10 × jaro_winkler
```

Weights are auto-normalized to sum to 1.0, so you can pass any relative values.

### Match Thresholds

| Score Range | Classification | Meaning |
|-------------|---------------|---------|
| ≥ 95 | `exact_match` | Almost certainly the same entity |
| 85 – 94 | `probable_match` | Very likely the same entity |
| 70 – 84 | `possible_match` | Might be the same, needs review |
| < 70 | `no_match` | Different entities |

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| **String metrics** | [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) 3.14.5 | High-performance Levenshtein, Token Sort/Set (C++ backend) |
| **Phonetics** | [Jellyfish](https://github.com/jamesturk/jellyfish) 1.2.1 | Jaro-Winkler, Soundex (Rust backend) |
| **Embeddings** | [fasttext-wheel](https://github.com/facebookresearch/fastText) 0.9.2 | 300-d subword word vectors for semantic similarity |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) 0.115.12 | Async REST API with auto-generated OpenAPI docs |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) 2.11.3 | Request/response schema validation |
| **Server** | [Uvicorn](https://www.uvicorn.org/) 0.34.3 | ASGI server |
| **Math** | [NumPy](https://numpy.org/) 2.2.6 | Vector operations (cosine similarity) |

## FastText Model Setup

The system uses Facebook's pre-trained FastText model `cc.en.300.bin` for embedding-based similarity. This model:
- Contains **300-dimensional vectors** for 2 million words
- Uses **subword (character n-gram) information** — can handle misspellings and OOV words
- File size: **~4.5 GB** (compressed ~1.5 GB)

### Download the Model

**Option A — Python script (recommended):**

```bash
# Activate your venv first
source venv/bin/activate

# Download to the project root
python -c "import fasttext.util; fasttext.util.download_model('en', if_exists='ignore')"
```

This downloads `cc.en.300.bin` to your current directory.

**Option B — Manual download:**

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

**Option C — Custom path:**

Set the `FASTTEXT_MODEL_PATH` environment variable:

```bash
export FASTTEXT_MODEL_PATH=/path/to/your/cc.en.300.bin
```

### What If I Don't Download the Model?

The system works fine without it — all 6 other metrics will still run. The embedding similarity will simply return `0.0` and the health endpoint will show `fasttext_model_loaded: false`.

## Installation

```bash
# 1. Clone / navigate to the project
cd job/

# 2. Create a virtual environment (Python 3.12)
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download the FastText model (~4.5 GB)
python -c "import fasttext.util; fasttext.util.download_model('en', if_exists='ignore')"
```

## Quick Start

```bash
# Start the server
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

API docs are auto-generated at: **http://localhost:8000/docs**

### Example: Compare Two Names

```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "string_a": "Abhinav K Baliyan",
    "string_b": "Abhinav Kumar Baliyan"
  }'
```

**Response:**

```json
{
  "string_a": "Abhinav K Baliyan",
  "string_b": "Abhinav Kumar Baliyan",
  "score": 87.45,
  "match_level": "probable_match",
  "features": {
    "levenshtein": 81.08,
    "jaro_winkler": 93.15,
    "jaccard": 66.67,
    "token_sort": 91.89,
    "token_set": 100.0,
    "soundex": 83.33,
    "embedding": 92.41
  }
}
```

### Example: Batch Compare

```bash
curl -X POST http://localhost:8000/api/v1/compare/batch \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {"string_a": "123 Main St", "string_b": "123 Main Street"},
      {"string_a": "Jon Smith", "string_b": "John Smith"},
      {"string_a": "HDFC Bank", "string_b": "ICICI Bank"}
    ]
  }'
```

### Example: Custom Weights

```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "string_a": "Robert Johnson",
    "string_b": "Bob Johnson",
    "weights": {
      "token_sort": 0.10,
      "embedding": 0.40,
      "levenshtein": 0.10,
      "token_set": 0.10,
      "jaccard": 0.10,
      "jaro_winkler": 0.10,
      "soundex": 0.10
    }
  }'
```

## API Reference

### `POST /api/v1/compare`

Compare a single pair of strings.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `string_a` | string | ✅ | First string |
| `string_b` | string | ✅ | Second string |
| `weights` | object | ❌ | Custom metric weights (auto-normalized) |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `string_a` | string | Echoed input |
| `string_b` | string | Echoed input |
| `score` | float | Final weighted score (0–100) |
| `match_level` | string | `exact_match` / `probable_match` / `possible_match` / `no_match` |
| `features` | object | Per-metric scores |

---

### `POST /api/v1/compare/batch`

Compare up to 1000 pairs in one request.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pairs` | array | ✅ | List of `{string_a, string_b, weights?}` objects |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | List of comparison results |
| `count` | int | Number of pairs compared |

---

### `GET /api/v1/health`

Health check and model status.

**Response:**

```json
{
  "status": "healthy",
  "fasttext_model_loaded": true,
  "available_metrics": ["levenshtein", "jaro_winkler", "jaccard", "token_sort", "token_set", "soundex", "embedding"]
}
```

## Project Structure

```
job/
├── venv/                  # Python 3.12 virtual environment
├── engine/                # Core similarity engine
│   ├── __init__.py
│   ├── normalizer.py      # Text normalization + tokenization
│   ├── similarity.py      # 6 similarity metric functions
│   ├── embeddings.py      # FastText model loading + cosine similarity
│   └── scorer.py          # Feature extraction, weighted scoring, classification
├── schemas.py             # Pydantic request/response models
├── main.py                # FastAPI application
├── requirements.txt       # Pinned dependencies
├── cc.en.300.bin          # FastText model (download separately)
└── README.md              # This file
```

## Customizing Weights

Pass custom weights in any request to adjust metric importance. All weights are auto-normalized:

```json
{
  "weights": {
    "embedding": 0.50,
    "token_sort": 0.30,
    "levenshtein": 0.20
  }
}
```

Only the metrics you specify will be used in scoring. This lets you tune the system for your specific use case (e.g., prioritize phonetic matching for names, or token matching for addresses).
