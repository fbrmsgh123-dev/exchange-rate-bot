"""환경변수 로드 및 검증.

다른 모듈은 os.environ을 직접 읽지 않고 반드시 get_config()를 통해 설정을 얻는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """환경변수가 없거나 값이 올바르지 않을 때."""


# PRD 가 지정한 대상 페이지. .env 의 SOURCE_URL 로 언제든 교체할 수 있다.
_DEFAULT_SOURCE_URL = (
    "https://exchange-rate-u7uio5wcnmmukgaqucpkml.streamlit.app/"
)
_DEFAULT_USER_AGENT = "ExchangeRateRAG/1.0"


@dataclass(frozen=True)
class Config:
    openai_api_key: str

    llm_model: str
    embedding_model: str

    source_url: str
    render_timeout_sec: int
    min_content_chars: int
    crawl_ttl_minutes: int
    user_agent: str
    enable_crawl: bool

    chunk_size: int
    chunk_overlap: int

    top_k: int
    similarity_threshold: float
    mmr_lambda: float
    retrieval_mode: str
    rrf_k: int
    bm25_min_ratio: float
    max_context_tokens: int
    max_history_turns: int

    vectorstore_dir: Path
    query_log_path: Path | None
    max_file_size_mb: int
    max_pages: int
    embed_batch_size: int


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"필수 환경변수 {name} 이(가) 설정되지 않았습니다. "
            f".env.example 을 참고해 .env 를 작성하세요."
        )
    return value


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 은(는) 정수여야 합니다 (현재 값: {raw!r}).") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} 은(는) true/false 여야 합니다 (현재 값: {raw!r}).")


def _get_optional_path(name: str, default: str) -> Path | None:
    """빈 문자열을 주면 비활성화한다."""
    raw = os.environ.get(name, default).strip()
    return Path(raw) if raw else None


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 은(는) 실수여야 합니다 (현재 값: {raw!r}).") from exc


@lru_cache(maxsize=1)
def get_config() -> Config:
    load_dotenv()

    cfg = Config(
        openai_api_key=_require("OPENAI_API_KEY"),
        llm_model=os.environ.get("LLM_MODEL", "gpt-5.6-sol").strip(),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-3-large"
        ).strip(),
        source_url=os.environ.get("SOURCE_URL", _DEFAULT_SOURCE_URL).strip(),
        render_timeout_sec=_get_int("RENDER_TIMEOUT_SEC", 30),
        min_content_chars=_get_int("MIN_CONTENT_CHARS", 200),
        crawl_ttl_minutes=_get_int("CRAWL_TTL_MINUTES", 10),
        user_agent=os.environ.get("USER_AGENT", _DEFAULT_USER_AGENT).strip(),
        enable_crawl=_get_bool("ENABLE_CRAWL", True),
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 200),
        top_k=_get_int("TOP_K", 5),
        similarity_threshold=_get_float("SIMILARITY_THRESHOLD", 0.3),
        mmr_lambda=_get_float("MMR_LAMBDA", 0.7),
        retrieval_mode=os.environ.get("RETRIEVAL_MODE", "vector").strip().lower(),
        rrf_k=_get_int("RRF_K", 10),
        bm25_min_ratio=_get_float("BM25_MIN_RATIO", 0.5),
        max_context_tokens=_get_int("MAX_CONTEXT_TOKENS", 4000),
        max_history_turns=_get_int("MAX_HISTORY_TURNS", 4),
        vectorstore_dir=Path(
            os.environ.get("VECTORSTORE_DIR", "./vectorstore").strip()
        ),
        query_log_path=_get_optional_path("QUERY_LOG_PATH", "./logs/queries.jsonl"),
        max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", 50),
        max_pages=_get_int("MAX_PAGES", 500),
        embed_batch_size=_get_int("EMBED_BATCH_SIZE", 100),
    )

    if cfg.chunk_size <= 0:
        raise ConfigError("CHUNK_SIZE 는 1 이상이어야 합니다.")
    if not 0 <= cfg.chunk_overlap < cfg.chunk_size:
        raise ConfigError(
            "CHUNK_OVERLAP 은 0 이상이면서 CHUNK_SIZE 보다 작아야 합니다 "
            f"(CHUNK_SIZE={cfg.chunk_size}, CHUNK_OVERLAP={cfg.chunk_overlap})."
        )
    if cfg.top_k <= 0:
        raise ConfigError("TOP_K 는 1 이상이어야 합니다.")
    if cfg.embed_batch_size <= 0:
        raise ConfigError("EMBED_BATCH_SIZE 는 1 이상이어야 합니다.")
    if not cfg.source_url.startswith(("http://", "https://")):
        raise ConfigError(
            f"SOURCE_URL 은 http:// 또는 https:// 로 시작해야 합니다 "
            f"(현재 값: {cfg.source_url!r})."
        )
    if cfg.render_timeout_sec <= 0:
        raise ConfigError("RENDER_TIMEOUT_SEC 는 1 이상이어야 합니다.")
    if cfg.min_content_chars <= 0:
        raise ConfigError("MIN_CONTENT_CHARS 는 1 이상이어야 합니다.")
    if cfg.crawl_ttl_minutes < 0:
        raise ConfigError("CRAWL_TTL_MINUTES 는 0 이상이어야 합니다.")
    if cfg.retrieval_mode not in ("hybrid", "vector"):
        raise ConfigError(
            "RETRIEVAL_MODE 는 hybrid 또는 vector 여야 합니다 "
            f"(현재 값: {cfg.retrieval_mode!r})."
        )
    if cfg.rrf_k <= 0:
        raise ConfigError("RRF_K 는 1 이상이어야 합니다.")
    if not 0.0 <= cfg.bm25_min_ratio <= 1.0:
        raise ConfigError(
            "BM25_MIN_RATIO 는 0.0~1.0 이어야 합니다 "
            f"(현재 값: {cfg.bm25_min_ratio})."
        )
    if not 0.0 <= cfg.mmr_lambda <= 1.0:
        raise ConfigError(
            "MMR_LAMBDA 는 0.0~1.0 이어야 합니다 "
            f"(1.0 = 다양성 무시, 관련도만 사용. 현재 값: {cfg.mmr_lambda})."
        )

    return cfg
