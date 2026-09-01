"""토큰 계산, 파일 해시, 로깅."""

from __future__ import annotations

import hashlib
import logging
import sys
from functools import lru_cache
from pathlib import Path

import tiktoken

_FALLBACK_ENCODING = "o200k_base"


def setup_logging(level: int = logging.INFO) -> None:
    # Windows 콘솔 기본 코드페이지에서 한글이 깨지는 것을 막는다
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@lru_cache(maxsize=8)
def _encoding_for(model: str):
    """모델 전용 인코딩이 없으면 최신 계열 기본 인코딩으로 대체한다.

    LLM_MODEL 은 tiktoken 이 모르는 값일 수 있으므로(PRD Q1) 실패를 허용하지 않는다.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def count_tokens(text: str, model: str) -> int:
    return len(_encoding_for(model).encode(text))


def file_hash(path: Path) -> str:
    """파일 내용의 SHA-256. 인덱스 캐시 키로 사용한다 (PRD F3.5)."""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_hash(data: bytes) -> str:
    """업로드된 바이트 스트림용 해시 (Streamlit 업로드 대비)."""
    return hashlib.sha256(data).hexdigest()
