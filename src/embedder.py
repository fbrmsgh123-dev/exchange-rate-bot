"""배치 임베딩 + 재시도 (PRD F3.1, F3.2, F3.6).

임베딩은 파이프라인에서 가장 비싼 단계다. 호출 횟수를 줄이기 위해 배치로 묶고,
일시적 실패는 지수 백오프로 흡수한다.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self, api_key: str, model: str, batch_size: int = 100) -> None:
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    model=self.model, input=list(texts)
                )
                return [item.embedding for item in response.data]
            except Exception as exc:  # 네트워크/레이트리밋/일시 오류
                last_exc = exc
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "임베딩 실패 (%d/%d), %.1f초 후 재시도: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise EmbeddingError(
            f"임베딩 API 호출이 {_MAX_RETRIES}회 모두 실패했습니다: {last_exc}"
        ) from last_exc

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """(n, dim) float32 배열을 반환한다. 순서는 입력과 동일하다."""
        if not texts:
            raise EmbeddingError("임베딩할 텍스트가 없습니다.")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
            done = min(start + self.batch_size, len(texts))
            logger.info("임베딩 진행 %d/%d", done, len(texts))
            if progress is not None:
                progress(done, len(texts))

        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """질의 단계용 단건 임베딩. (dim,) 배열을 반환한다."""
        return self.embed_texts([text])[0]
