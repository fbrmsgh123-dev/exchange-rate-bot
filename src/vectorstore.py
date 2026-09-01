"""FAISS 인덱스 생성/저장/로드/검색 (PRD F3.3~F3.5).

IndexFlatIP + L2 정규화 = 코사인 유사도. 정규화는 **인덱싱과 질의 양쪽 모두**에서
수행해야 하며, 한쪽이라도 빠지면 유사도 점수와 SIMILARITY_THRESHOLD 가 무의미해진다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE = "meta.json"


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    index: int = -1  # FAISS 내부 위치. MMR 이 벡터를 되찾을 때 쓴다


def _normalized(vectors: np.ndarray) -> np.ndarray:
    """복사본을 L2 정규화해 반환한다 (원본 훼손 방지)."""
    # ascontiguousarray 는 조건이 맞으면 원본을 그대로 돌려주므로,
    # normalize_L2 가 in-place 로 호출자의 배열을 훼손하지 않도록 반드시 복사한다.
    matrix = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    faiss.normalize_L2(matrix)
    return matrix


def store_dir(vectorstore_dir: Path, doc_hash: str) -> Path:
    """문서 해시별 캐시 경로 (PRD F3.5)."""
    return Path(vectorstore_dir) / doc_hash


class VectorStore:
    def __init__(
        self,
        index: faiss.Index,
        chunks: list[Chunk],
        meta: dict,
    ) -> None:
        if index.ntotal != len(chunks):
            raise VectorStoreError(
                f"인덱스 벡터 수({index.ntotal})와 청크 수({len(chunks)})가 다릅니다."
            )
        self._index = index
        self.chunks = chunks
        self.meta = meta

    @property
    def size(self) -> int:
        return self._index.ntotal

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        *,
        doc_hash: str,
        embedding_model: str,
        extra_meta: dict | None = None,
    ) -> VectorStore:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("청크 수와 임베딩 수가 일치하지 않습니다.")

        matrix = _normalized(embeddings)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        meta = {
            "doc_hash": doc_hash,
            "embedding_model": embedding_model,
            "dimension": int(matrix.shape[1]),
            "chunk_count": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(extra_meta or {}),
        }
        logger.info(
            "FAISS 인덱스 생성: %d벡터 x %d차원", index.ntotal, matrix.shape[1]
        )
        return cls(index, chunks, meta)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILE))
        self.save_meta(path)
        logger.info("인덱스 저장 완료: %s", path)

    def save_meta(self, path: Path) -> None:
        """벡터는 그대로 두고 메타데이터만 다시 쓴다.

        본문이 안 바뀐 재수집에서 `crawled_at` 만 갱신하는 경로(PRD F4.5)가 이걸 쓴다.
        이 경로는 TTL 마다 반복 실행되므로 벡터 파일을 다시 쓰면 순수한 낭비다.
        """
        path.mkdir(parents=True, exist_ok=True)
        payload = {**self.meta, "chunks": [c.to_dict() for c in self.chunks]}
        (path / _META_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, *, embedding_model: str) -> VectorStore:
        """저장된 인덱스를 읽는다.

        임베딩 모델이 바뀌면 기존 벡터는 전부 무효이므로 로드를 거부한다.
        """
        index_path = path / _INDEX_FILE
        meta_path = path / _META_FILE
        if not index_path.exists() or not meta_path.exists():
            raise VectorStoreError(f"인덱스가 없습니다: {path}")

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorStoreError(f"인덱스를 읽을 수 없습니다: {exc}") from exc

        stored_model = payload.get("embedding_model")
        if stored_model != embedding_model:
            raise VectorStoreError(
                f"임베딩 모델이 다릅니다 (저장됨: {stored_model}, 현재: {embedding_model}). "
                f"인덱스를 재생성해야 합니다."
            )

        chunk_dicts = payload.pop("chunks", [])
        # from_dict 로만 복원한다 — 스키마가 달라진 인덱스를 만나도 죽지 않는다.
        chunks = [Chunk.from_dict(c) for c in chunk_dicts]
        return cls(index, chunks, payload)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        """코사인 유사도 상위 top_k 를 반환한다 (점수 내림차순)."""
        if self.size == 0:
            return []
        query = _normalized(query_vector)
        scores, indices = self._index.search(query, min(top_k, self.size))

        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS 는 결과 부족 시 -1 을 채운다
                continue
            hits.append(
                SearchHit(
                    chunk=self.chunks[int(idx)],
                    score=float(score),
                    index=int(idx),
                )
            )
        return hits

    def reconstruct(self, index: int) -> np.ndarray:
        """저장된 (정규화된) 벡터를 되찾는다.

        MMR 은 후보끼리의 유사도를 알아야 하는데, 다시 임베딩하면 API 를 또 호출하는
        셈이다. `IndexFlatIP` 는 원본 벡터를 그대로 들고 있으므로 여기서 꺼내 쓴다.
        """
        return self._index.reconstruct(int(index))
