"""PDF -> 페이지별 텍스트 (PRD F1)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

# 페이지당 이 글자 수 미만이면 텍스트 레이어가 없는 것으로 간주 (PRD F1.4)
_SCANNED_CHARS_PER_PAGE = 30


class PDFLoadError(RuntimeError):
    """PDF 를 읽을 수 없을 때 사용자에게 그대로 보여줄 수 있는 메시지."""


@dataclass(frozen=True)
class Page:
    page: int  # 1-based, 사용자에게 보이는 페이지 번호
    text: str


def _normalize(text: str) -> str:
    """공백/개행 정규화 (PRD F2.4)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def load_pdf(path: Path, *, max_file_size_mb: int, max_pages: int) -> list[Page]:
    """PDF 를 페이지 단위로 읽는다. 페이지 번호는 이후 출처 인용의 근거가 된다."""
    if not path.exists():
        raise PDFLoadError(f"파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFLoadError("PDF 파일만 지원합니다.")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_file_size_mb:
        raise PDFLoadError(
            f"파일이 너무 큽니다 ({size_mb:.1f}MB). 최대 {max_file_size_mb}MB 까지 지원합니다."
        )

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF 는 다양한 예외를 던진다
        raise PDFLoadError(f"PDF 를 열 수 없습니다: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise PDFLoadError("암호가 걸린 PDF 는 지원하지 않습니다.")
        if doc.page_count > max_pages:
            raise PDFLoadError(
                f"페이지가 너무 많습니다 ({doc.page_count}p). 최대 {max_pages}p 까지 지원합니다."
            )

        pages = [
            Page(page=i + 1, text=_normalize(doc[i].get_text("text")))
            for i in range(doc.page_count)
        ]

    total_chars = sum(len(p.text) for p in pages)
    if pages and total_chars < _SCANNED_CHARS_PER_PAGE * len(pages):
        raise PDFLoadError(
            "텍스트를 추출할 수 없습니다. OCR 이 필요한 스캔 문서로 보입니다."
        )

    non_empty = sum(1 for p in pages if p.text)
    logger.info(
        "%s: %d 페이지 중 %d 페이지에서 텍스트 추출 (%d자)",
        path.name,
        len(pages),
        non_empty,
        total_chars,
    )
    return pages
