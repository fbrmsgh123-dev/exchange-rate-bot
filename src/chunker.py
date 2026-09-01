"""텍스트 -> 청크 + 메타데이터 (PRD F2).

두 개의 입구가 있다.

- `chunk_blocks()` (웹, 현행): 수집기가 만든 의미 블록을 청크로 옮긴다.
  **블록이 chunk_size 이하면 쪼개지 않는다** — 수집기가 이미 의미 단위로 묶어 둔 것을
  (PRD F2.8) 여기서 다시 자르면 통화 카드가 반쪽만 남는 청크가 생긴다.
- `chunk_pages()` (PDF, 레거시): 청킹을 **페이지 단위로** 수행한다. 청크가 페이지 경계를
  넘지 않아야 [p.N] 출처를 정확히 붙일 수 있기 때문이다.

두 경우 모두 출처 메타데이터는 이 단계에서 부착해 끝까지 보존한다 (PRD F2.3).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from urllib.parse import urlparse

from .pdf_loader import Page
from .web_loader import Block

logger = logging.getLogger(__name__)

# 우선순위: 문단 -> 줄 -> 문장 -> 공백 -> 글자 (PRD F2.1)
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# 전체 페이지의 이 비율 이상에서 반복되는 짧은 줄은 머리말/꼬리말로 본다 (PRD F2.4)
_REPEAT_RATIO = 0.5
_HEADER_MAX_CHARS = 80


@dataclass(frozen=True)
class Chunk:
    """검색 단위. **이 클래스는 곧 `meta.json` 의 직렬화 스키마다.**

    저장은 `to_dict()`, 복원은 `from_dict()` 로만 한다. 새 필드는 반드시 기본값을
    주어 추가할 것 — 기본값 없이 추가하면 이미 저장된 모든 인덱스의 로드가 깨진다.
    반대로 `from_dict()` 는 모르는 키를 무시하므로, 나중 버전이 저장한 인덱스를
    옛 코드가 읽어도 죽지 않는다.
    """

    chunk_id: int
    source_file: str  # 웹: 호스트명 / PDF: 파일명
    page: int  # PDF 의 1-based 페이지. 웹 소스에서는 0
    text: str

    # --- 웹 소스용 출처 메타데이터 (PRD F2.3) ---
    block_id: int = -1
    section: str = ""
    block_type: str = ""
    source_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> Chunk:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


def _merge_splits(
    splits: list[str], separator: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    sep_len = len(separator)
    chunks: list[str] = []
    current: list[str] = []
    total = 0

    for piece in splits:
        piece_len = len(piece)
        addition = piece_len + (sep_len if current else 0)

        if current and total + addition > chunk_size:
            chunks.append(separator.join(current).strip())
            # overlap 만큼만 남기고 앞에서부터 버린다
            while current and (
                total > chunk_overlap or total + addition > chunk_size
            ):
                total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                current.pop(0)

        current.append(piece)
        total += piece_len + (sep_len if len(current) > 1 else 0)

    if current:
        chunks.append(separator.join(current).strip())

    return [c for c in chunks if c]


def _split_text(
    text: str, separators: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    final: list[str] = []
    separator = separators[-1]
    remaining: list[str] = []

    for i, candidate in enumerate(separators):
        if candidate == "":
            separator = candidate
            break
        if candidate in text:
            separator = candidate
            remaining = separators[i + 1 :]
            break

    splits = list(text) if separator == "" else text.split(separator)
    splits = [s for s in splits if s]

    buffer: list[str] = []
    for piece in splits:
        if len(piece) <= chunk_size:
            buffer.append(piece)
            continue
        if buffer:
            final.extend(_merge_splits(buffer, separator, chunk_size, chunk_overlap))
            buffer = []
        if remaining:
            final.extend(_split_text(piece, remaining, chunk_size, chunk_overlap))
        else:
            final.append(piece)

    if buffer:
        final.extend(_merge_splits(buffer, separator, chunk_size, chunk_overlap))

    return final


def _strip_repeated_lines(pages: list[Page]) -> list[Page]:
    """여러 페이지에 반복 등장하는 짧은 줄(머리말/꼬리말)을 제거한다."""
    if len(pages) < 4:
        return pages

    counter: Counter[str] = Counter()
    for page in pages:
        lines = {
            line.strip()
            for line in page.text.split("\n")
            if 0 < len(line.strip()) <= _HEADER_MAX_CHARS
        }
        counter.update(lines)

    threshold = max(3, int(len(pages) * _REPEAT_RATIO))
    repeated = {
        line
        for line, count in counter.items()
        if count >= threshold and not re.fullmatch(r"[\d\s.\-]+", line)
    }
    if not repeated:
        return pages

    logger.info("머리말/꼬리말로 판단해 제거한 줄: %d종", len(repeated))
    cleaned = []
    for page in pages:
        kept = [
            line for line in page.text.split("\n") if line.strip() not in repeated
        ]
        cleaned.append(Page(page=page.page, text="\n".join(kept).strip()))
    return cleaned


def chunk_pages(
    pages: list[Page],
    *,
    source_file: str,
    chunk_size: int,
    chunk_overlap: int,
    start_id: int = 0,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_id = start_id

    for page in _strip_repeated_lines(pages):
        if not page.text:
            continue
        for text in _split_text(page.text, _SEPARATORS, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=next_id,
                    source_file=source_file,
                    page=page.page,
                    text=text,
                )
            )
            next_id += 1

    logger.info("%s: %d개 청크 생성", source_file, len(chunks))
    return chunks


def _host_of(url: str) -> str:
    return urlparse(url).netloc or url


def chunk_blocks(
    blocks: Sequence[Block],
    *,
    source_url: str,
    chunk_size: int,
    chunk_overlap: int,
    start_id: int = 0,
) -> list[Chunk]:
    """의미 블록 -> 청크 (PRD F2.4, F2.8).

    블록이 `chunk_size` 이하면 **그대로 청크 하나가 된다.** 수집기가 Streamlit 요소
    경계로 묶어 둔 단위를 유지해야, 청크 하나만 검색돼도 어느 통화의 값인지 알 수 있다.
    `chunk_size` 를 넘는 블록(예: 뉴스 묶음)만 재귀 분할하고, 쪼개진 조각은 원래
    블록의 메타데이터를 그대로 물려받는다.
    """
    chunks: list[Chunk] = []
    next_id = start_id
    host = _host_of(source_url)
    split_count = 0

    # 소제목 블록은 색인하지 않는다. 소제목 텍스트는 뒤따르는 블록들의 `section`
    # 메타데이터로 이미 보존되므로, 따로 청크를 만들면 내용 없는 청크가 Top-K 한
    # 자리를 차지해 정작 값이 담긴 청크를 밀어낸다. 단, 소제목뿐인 페이지에서
    # 아무것도 색인하지 않는 사고를 막기 위해 본문 블록이 하나라도 있을 때만 건너뛴다.
    skip_headings = any(block.block_type != "heading" for block in blocks)

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if skip_headings and block.block_type == "heading":
            continue

        if len(text) <= chunk_size:
            parts = [text]
        else:
            parts = _split_text(text, _SEPARATORS, chunk_size, chunk_overlap)
            split_count += 1

        for part in parts:
            part = part.strip()
            if not part:
                continue
            chunks.append(
                Chunk(
                    chunk_id=next_id,
                    source_file=host,
                    page=0,  # 웹 소스에는 페이지 개념이 없다
                    text=part,
                    block_id=block.block_id,
                    section=block.section,
                    block_type=block.block_type,
                    source_url=source_url,
                )
            )
            next_id += 1

    logger.info(
        "%s: 블록 %d개 -> 청크 %d개 (분할된 블록 %d개)",
        host,
        len(blocks),
        len(chunks),
        split_count,
    )
    return chunks
