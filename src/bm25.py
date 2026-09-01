"""BM25 어휘 검색 (PRD F5.9).

벡터 검색이 못 하는 일을 메운다: **질문에 페이지의 단어가 그대로 들어 있을 때**
확실히 집는 것. 임베딩은 의미를 잘 잡지만 "JPY", "신저가", "종부세" 같은 짧고
구체적인 토큰에서는 희석된다.

반대로 BM25 가 못 하는 일도 분명하다 — **표기가 다르면 못 잡는다.** "엔화" 와
"일본 옌" 은 글자가 겹치지 않으므로 BM25 로는 연결되지 않는다. 그쪽은 여전히
`retriever._CURRENCY_ALIASES` 가 담당한다. 둘은 대체 관계가 아니라 보완 관계다.

## 한국어 토큰화

형태소 분석기(konlpy/mecab)를 도입하지 않았다. 무거운 의존성 하나를 들이는 대가로
얻는 게, 이 규모(청크 10여 개)에서는 크지 않다. 대신:

- 한글은 **2-gram 음절 시프트**로 자른다. 조사가 붙어도 겹치게 하려는 것이다
  ("엔화는" -> {엔화, 화는} vs "엔화" -> {엔화}).
- 영숫자는 토큰을 그대로 둔다 ("USD", "JPY" 가 쪼개지면 오히려 나빠진다).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_K1 = 1.5
_B = 0.75

_HANGUL_RUN = re.compile(r"[가-힣]+")
_ALNUM_RUN = re.compile(r"[a-z0-9]+")
_BIGRAM_MIN = 2


def tokenize(text: str) -> list[str]:
    """한글은 2-gram, 영숫자는 원형으로 자른다."""
    lowered = text.lower()
    tokens: list[str] = []

    for run in _HANGUL_RUN.findall(lowered):
        if len(run) < _BIGRAM_MIN:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + _BIGRAM_MIN] for i in range(len(run) - 1))

    tokens.extend(_ALNUM_RUN.findall(lowered))
    return tokens


@dataclass(frozen=True)
class BM25Hit:
    index: int
    score: float
    matched: tuple[str, ...]  # 어떤 토큰이 맞았는지 (디버깅·설명용)


class BM25Index:
    """Okapi BM25. 청크 수가 적어 매번 메모리에서 만들어도 무시할 만하다."""

    def __init__(self, documents: Sequence[str]) -> None:
        self._docs = [tokenize(doc) for doc in documents]
        self._lengths = [len(doc) for doc in self._docs]
        self._counts = [Counter(doc) for doc in self._docs]
        self._avg_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

        document_frequency: Counter[str] = Counter()
        for tokens in self._counts:
            document_frequency.update(tokens.keys())

        total = len(self._docs)
        # 모든 문서에 나오는 토큰은 IDF 가 0 에 가까워 자연히 무시된다.
        self._idf = {
            token: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    @property
    def size(self) -> int:
        return len(self._docs)

    def search(self, query: str, *, min_score: float = 0.0) -> list[BM25Hit]:
        """점수 내림차순. `min_score` 미만은 버린다."""
        query_tokens = set(tokenize(query))
        if not query_tokens or not self._docs:
            return []

        hits: list[BM25Hit] = []
        for index, counts in enumerate(self._counts):
            score = 0.0
            matched: list[str] = []
            length = self._lengths[index] or 1

            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                idf = self._idf.get(token, 0.0)
                if idf <= 0.0:
                    continue
                denominator = frequency + _K1 * (
                    1 - _B + _B * length / (self._avg_length or 1)
                )
                score += idf * frequency * (_K1 + 1) / denominator
                matched.append(token)

            if score >= min_score and score > 0.0:
                hits.append(
                    BM25Hit(index=index, score=score, matched=tuple(sorted(matched)))
                )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits
