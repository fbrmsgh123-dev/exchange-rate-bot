"""인덱싱 파이프라인: 소스 -> 청킹 -> 임베딩 -> FAISS 저장.

두 개의 입구가 있다.

- `ingest_url()` (웹, 현행): 렌더링 -> 블록 -> 청킹 -> 임베딩 -> 저장.
- `ingest()` (PDF, 레거시): 파일 파싱 -> 페이지 청킹 -> 임베딩 -> 저장.

임베딩이 파이프라인에서 가장 비싼 단계이므로 **캐시 판정이 이 모듈의 핵심**이다.
웹 경로의 캐시 키 구조는 두 층이다:

- 디렉터리 이름 = `url_hash` — URL 하나당 디렉터리 하나. 본문 해시를 디렉터리
  이름으로 쓰면 환율이 바뀔 때마다 새 디렉터리가 쌓여 디스크가 무한정 늘어난다.
- 재임베딩 여부 = `meta["content_hash"]` — 본문이 그대로면 임베딩을 건너뛴다 (PRD F4.5).

질의 단계(M3)와 섞지 말 것. Streamlit UI(M5)도 이 함수를 그대로 호출한다.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config import Config

from .chunker import Chunk, chunk_blocks, chunk_pages
from .embedder import Embedder
from .utils import file_hash
from .vectorstore import VectorStore, VectorStoreError, store_dir
from .web_loader import PageSnapshot, WebLoadError, load_url

logger = logging.getLogger(__name__)

_SNAPSHOT_HTML = "snapshot.html"
_SNAPSHOT_TEXT = "snapshot.txt"


@dataclass(frozen=True)
class IngestResult:
    store: VectorStore
    path: Path
    doc_hash: str
    from_cache: bool
    chunk_count: int

    # PDF 경로용
    page_count: int = 0

    # 웹 경로용
    content_hash: str = ""
    crawled_at: str = ""
    block_count: int = 0
    source_url: str = ""
    stale: bool = False  # 수집 실패로 이전 스냅샷을 재사용했는가 (PRD F1.6)
    warnings: tuple[str, ...] = ()


def corpus_hash(paths: list[Path]) -> str:
    """여러 PDF를 하나의 인덱스로 묶을 때의 캐시 키.

    파일 순서가 달라도 같은 해시가 나오도록 정렬해 결합한다.
    """
    parts = sorted(file_hash(p) for p in paths)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def build_chunks(paths: list[Path], cfg: Config) -> tuple[list[Chunk], int]:
    # PDF 경로는 레거시다. 여기서 지연 import 해야 PyMuPDF 가 없는 배포
    # 환경에서도 웹 경로가 동작한다.
    from .pdf_loader import load_pdf

    chunks: list[Chunk] = []
    page_total = 0
    for path in paths:
        pages = load_pdf(
            path, max_file_size_mb=cfg.max_file_size_mb, max_pages=cfg.max_pages
        )
        page_total += len(pages)
        chunks.extend(
            chunk_pages(
                pages,
                source_file=path.name,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                start_id=len(chunks),
            )
        )
    if not chunks:
        raise ValueError("추출된 텍스트가 없어 인덱스를 만들 수 없습니다.")
    return chunks, page_total


def ingest(
    paths: list[Path],
    cfg: Config,
    *,
    rebuild: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> IngestResult:
    doc_hash = corpus_hash(paths)
    path = store_dir(cfg.vectorstore_dir, doc_hash)

    if not rebuild and (path / "index.faiss").exists():
        try:
            store = VectorStore.load(path, embedding_model=cfg.embedding_model)
            logger.info("캐시된 인덱스 재사용 (임베딩 비용 0): %s", path)
            return IngestResult(
                store=store,
                path=path,
                doc_hash=doc_hash,
                from_cache=True,
                page_count=int(store.meta.get("page_count", 0)),
                chunk_count=store.size,
            )
        except Exception as exc:
            # 손상되거나 모델이 바뀐 인덱스는 재생성한다 (PRD §8)
            logger.warning("기존 인덱스를 쓸 수 없어 재생성합니다: %s", exc)

    chunks, page_count = build_chunks(paths, cfg)

    embedder = Embedder(
        api_key=cfg.openai_api_key,
        model=cfg.embedding_model,
        batch_size=cfg.embed_batch_size,
    )
    embeddings = embedder.embed_texts([c.text for c in chunks], progress=progress)

    store = VectorStore.build(
        chunks,
        embeddings,
        doc_hash=doc_hash,
        embedding_model=cfg.embedding_model,
        extra_meta={
            "files": [p.name for p in paths],
            "page_count": page_count,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
        },
    )
    store.save(path)

    return IngestResult(
        store=store,
        path=path,
        doc_hash=doc_hash,
        from_cache=False,
        page_count=page_count,
        chunk_count=len(chunks),
    )


# --------------------------------------------------------------------------- #
# 웹 경로 (PRD F1~F4)
# --------------------------------------------------------------------------- #


def url_hash(url: str) -> str:
    """URL 하나당 인덱스 디렉터리 하나 (PRD F3.4)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def build_url_chunks(snapshot: PageSnapshot, cfg: Config) -> list[Chunk]:
    chunks = chunk_blocks(
        snapshot.blocks,
        source_url=snapshot.url,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    if not chunks:
        raise ValueError("추출된 텍스트가 없어 인덱스를 만들 수 없습니다.")
    return chunks


def _save_snapshot(path: Path, snapshot: PageSnapshot) -> None:
    """원본 스냅샷 보존 (PRD F1.7).

    브라우저 없이 파서만 다시 돌려볼 수 있어야 하고(`web_loader.load_html()`),
    DOM 구조가 바뀌어 추출이 깨졌을 때 대조할 원본이 필요하다.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / _SNAPSHOT_HTML).write_text(snapshot.html, encoding="utf-8")
    (path / _SNAPSHOT_TEXT).write_text(snapshot.content_text, encoding="utf-8")


def _load_existing(path: Path, cfg: Config) -> VectorStore | None:
    if not (path / "index.faiss").exists():
        return None
    try:
        return VectorStore.load(path, embedding_model=cfg.embedding_model)
    except VectorStoreError as exc:
        # 손상되었거나 임베딩 모델이 바뀐 인덱스는 버리고 재생성한다 (PRD §8)
        logger.warning("기존 인덱스를 쓸 수 없어 재생성합니다: %s", exc)
        return None


def ingest_url(
    cfg: Config,
    *,
    url: str | None = None,
    rebuild: bool = False,
    respect_robots: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> IngestResult:
    """URL 을 수집해 인덱스를 만들거나 재사용한다.

    반환값의 `from_cache` 는 "임베딩 API 를 호출하지 않았다"는 뜻이고,
    `stale` 은 "수집이 실패해 이전 데이터로 답하고 있다"는 뜻이다 (PRD F1.6).
    """
    url = url or cfg.source_url
    path = store_dir(cfg.vectorstore_dir, url_hash(url))
    existing = _load_existing(path, cfg)

    try:
        snapshot = load_url(
            url,
            timeout_sec=cfg.render_timeout_sec,
            min_content_chars=cfg.min_content_chars,
            user_agent=cfg.user_agent,
            respect_robots=respect_robots,
        )
    except WebLoadError as exc:
        # 수집 실패. 쓸 수 있는 인덱스가 있으면 그것으로 서비스를 이어간다.
        if existing is None:
            raise
        logger.warning("수집 실패로 이전 인덱스를 사용합니다: %s", exc)
        return IngestResult(
            store=existing,
            path=path,
            doc_hash=url_hash(url),
            from_cache=True,
            chunk_count=existing.size,
            content_hash=str(existing.meta.get("content_hash", "")),
            crawled_at=str(existing.meta.get("crawled_at", "")),
            block_count=int(existing.meta.get("block_count", 0)),
            source_url=url,
            stale=True,
            warnings=(str(exc),),
        )

    # 본문이 그대로면 재임베딩하지 않는다 (PRD F4.5). crawled_at 만 갱신한다.
    if (
        not rebuild
        and existing is not None
        and existing.meta.get("content_hash") == snapshot.content_hash
    ):
        existing.meta["crawled_at"] = snapshot.crawled_at
        existing.save_meta(path)  # 벡터는 그대로 — 메타만 갱신
        _save_snapshot(path, snapshot)
        logger.info("본문 변경 없음 — 재임베딩 생략 (임베딩 비용 0)")
        return IngestResult(
            store=existing,
            path=path,
            doc_hash=url_hash(url),
            from_cache=True,
            chunk_count=existing.size,
            content_hash=snapshot.content_hash,
            crawled_at=snapshot.crawled_at,
            block_count=len(snapshot.blocks),
            source_url=url,
            warnings=snapshot.warnings,
        )

    chunks = build_url_chunks(snapshot, cfg)

    embedder = Embedder(
        api_key=cfg.openai_api_key,
        model=cfg.embedding_model,
        batch_size=cfg.embed_batch_size,
    )
    embeddings = embedder.embed_texts([c.text for c in chunks], progress=progress)

    store = VectorStore.build(
        chunks,
        embeddings,
        doc_hash=url_hash(url),
        embedding_model=cfg.embedding_model,
        extra_meta={
            "source_url": snapshot.url,
            "final_url": snapshot.final_url,
            "frame_url": snapshot.frame_url,
            "content_hash": snapshot.content_hash,
            "crawled_at": snapshot.crawled_at,
            "block_count": len(snapshot.blocks),
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
        },
    )
    store.save(path)
    _save_snapshot(path, snapshot)

    return IngestResult(
        store=store,
        path=path,
        doc_hash=url_hash(url),
        from_cache=False,
        chunk_count=len(chunks),
        content_hash=snapshot.content_hash,
        crawled_at=snapshot.crawled_at,
        block_count=len(snapshot.blocks),
        source_url=url,
        warnings=snapshot.warnings,
    )
