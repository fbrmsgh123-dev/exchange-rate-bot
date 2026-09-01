"""질의 -> Top-K 청크 (PRD F5).

**이 모듈이 토큰 효율의 핵심이다.** 질문마다 실행되므로 저렴해야 하고, 여기서
걸러낸 만큼만 LLM 프롬프트에 들어간다. 임베딩 API 호출은 질문당 정확히 1회다.

순서: 별칭 확장 -> 질문 임베딩 -> (벡터 검색 + BM25 검색) -> RRF 융합 -> MMR -> 토큰 상한.

`RETRIEVAL_MODE=vector` 로 두면 BM25 를 끄고 벡터만 쓴다.

검색 결과가 0건이면 **호출자는 LLM 을 호출하지 않는다** (PRD F5.5). 이 판정을
여기서 하고 `RetrievalResult.is_empty` 로 알려 준다.

## 하이브리드 설계 근거 (실측, 2026-09-01)

- **BM25 절대 점수로는 관련/무관을 가를 수 없다.** "달러 환율 얼마야?"(2.419)와
  "미국 드라마 추천해줘"(2.419)가 같은 점수다. 둘 다 흔한 토큰 하나가 맞았을 뿐이다.
  그래서 BM25 점수에 의미 있는 문턱을 세우려는 시도는 포기했다.
- **대신 그 위험은 이미 벡터 쪽에 있다.** "미국 드라마 추천해줘"는 벡터 유사도
  0.499 로 임계값 0.3 을 이미 통과한다. BM25 를 넣어서 새로 생기는 문제가 아니다.
- **BM25 만이 잡는 것이 실제로 있다.** "특이사항 있어?" 는 벡터 최고 0.234 로 0건
  이었지만, 섹션 라벨을 BM25 문서에 포함하면 잡힌다. 이게 하이브리드의 실이득이다.
- 완전히 무관한 질문(김치찌개·파이썬·날씨)은 BM25 점수가 **정확히 0** 이다. IDF 가
  흔한 토큰을 0 으로 눌러 주므로, 0 초과만 후보로 받아도 0건 조기 반환은 지켜진다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from config import Config

from .bm25 import BM25Index
from .embedder import Embedder
from .utils import count_tokens
from .vectorstore import SearchHit, VectorStore

logger = logging.getLogger(__name__)

# 임계값을 넘은 후보를 MMR 에 넘기기 전에 이 배수만큼 넉넉히 뽑는다.
# Top-K 만 뽑아 놓고 MMR 을 돌리면 고를 여지가 없어 다양성 확보가 무의미해진다.
_CANDIDATE_MULTIPLIER = 3

# 통화 별칭 (PRD F5.8 — Must).
#
# 순수 벡터 검색은 짧은 질의에 약하다. 실측: "엔화는?" 으로 검색하면 유로 카드가
# 1위(0.378), 정작 일본 카드는 3위(0.365)로 밀렸다. 대상 페이지가 "엔" 이 아니라
# **"일본 옌"** 으로 표기하기 때문이다. 질의에 페이지의 표기를 덧붙여 이를 보정한다.
#
# 트리거는 부분 문자열로 맞춘다("엔화는?" 처럼 조사가 붙어도 걸리게). "엔진" 같은
# 오탐이 가능하지만, 그때 붙는 것은 검색어 몇 단어뿐이고 무관한 질문은 어차피
# 임계값에서 걸러진다 — 순위가 뒤집히는 오답보다 훨씬 싼 비용이다.
_CURRENCY_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("달러", "dollar", "usd", "미국"), "미국 달러 USD"),
    (("유로", "euro", "eur"), "유로 EUR"),
    (("엔화", "엔", "옌", "jpy", "yen", "일본"), "일본 옌 JPY"),
    (("위안", "yuan", "cny", "인민폐", "런민비", "중국"), "중국 위안 CNY"),
)

# 후속 질문 판정 (PRD F5.7).
#
# **"짧으면 후속 질문" 으로 판정하면 안 된다.** M5 실측: 대화 중에 "김치찌개 레시피
# 알려줘" 를 물으면 짧다는 이유로 직전 질문("달러 환율 얼마야?")이 앞에 붙어,
# 달러 카드가 임계값을 넘어버렸다. 무관한 질문인데도 0건 조기 반환(F5.5)이 발동하지
# 않고 LLM 을 호출해 836 토큰을 썼다.
#
# 그래서 **명시적인 지시어가 있을 때만** 맥락을 끌어온다. 지시어가 없으면 질문을
# 그 자체로 판단한다 — 판정이 틀렸을 때 "찾을 수 없음" 이 나오는 쪽이, 엉뚱한
# 주제의 근거로 답하는 쪽보다 안전하다.
_FOLLOWUP_MARKERS = (
    "그럼",
    "그러면",
    "그건",
    "그거",
    "그것",
    "저건",
    "저거",
    "저것",
    "이건",
    "이거",
    "이것",
    "나머지",
    "반대로",
    "다음은",
    "그리고",
)
_FOLLOWUP_MAX_CHARS = 20


@dataclass(frozen=True)
class RetrievalResult:
    hits: tuple[SearchHit, ...]
    query: str  # 실제로 임베딩한 확장 질의 (디버깅용)
    context_tokens: int
    dropped_by_threshold: int = 0
    dropped_by_token_cap: int = 0
    best_score: float = 0.0
    mode: str = "vector"
    # BM25 만 찾아낸 청크의 block_id. 하이브리드가 실제로 기여했는지 보는 창이다.
    lexical_only: tuple[int, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True 면 LLM 을 호출하지 않는다 (PRD F5.5)."""
        return not self.hits

    def sections(self) -> tuple[str, ...]:
        seen: list[str] = []
        for hit in self.hits:
            if hit.chunk.section and hit.chunk.section not in seen:
                seen.append(hit.chunk.section)
        return tuple(seen)


def expand_query(question: str, history: Sequence[tuple[str, str]] = ()) -> str:
    """질의를 검색용으로 확장한다 (PRD F5.7, F5.8).

    `history` 는 `(질문, 답변)` 튜플의 순서 있는 목록이다.
    """
    lowered = question.lower()
    extras = [
        canonical
        for triggers, canonical in _CURRENCY_ALIASES
        if any(trigger in lowered for trigger in triggers)
    ]

    # 지시어가 들어간 짧은 질문("그럼 저건?")은 그 자체로는 검색이 불가능하다.
    # 직전 질문을 덧붙여 맥락을 복원한다. LLM 재작성(추가 호출)을 피한 휴리스틱이다.
    stripped = question.strip()
    is_followup = (
        not extras
        and bool(history)
        and len(stripped) <= _FOLLOWUP_MAX_CHARS
        and any(marker in stripped for marker in _FOLLOWUP_MARKERS)
    )
    if is_followup:
        previous = history[-1][0].strip()
        if previous:
            logger.debug("후속 질문으로 판단 — 직전 질문을 검색어에 합침: %r", previous)
            return f"{previous} {question}"

    if not extras:
        return question
    return f"{question} ({' '.join(extras)})"


def _rrf_fuse(
    vector_order: list[int], lexical_order: list[int], *, rrf_k: int
) -> dict[int, float]:
    """Reciprocal Rank Fusion — 두 순위를 합친다.

    점수를 정규화해 더하지 않는 이유: 코사인(0~1)과 BM25(상한 없음)는 스케일이
    비교 불가능하다. RRF 는 **순위만** 쓰므로 그 문제를 피하고, 두 목록에 모두
    등장한 청크에 자연히 가중을 준다.
    """
    fused: dict[int, float] = {}
    for order in (vector_order, lexical_order):
        for rank, index in enumerate(order):
            fused[index] = fused.get(index, 0.0) + 1.0 / (rrf_k + rank + 1)
    return fused


def _mmr_select(
    store: VectorStore,
    candidates: list[SearchHit],
    *,
    top_k: int,
    lambda_: float,
    relevance: dict[int, float] | None = None,
) -> list[SearchHit]:
    """관련도와 다양성을 함께 보는 선택 (PRD F5.6).

    같은 내용을 담은 청크가 Top-K 를 다 차지하면 컨텍스트가 낭비된다.
    `lambda_ = 1.0` 이면 MMR 을 끈 것과 같다(관련도 순서 그대로).

    `relevance` 를 주면 관련도로 그 값을 쓴다(하이브리드에서는 융합 점수). 주지
    않으면 코사인 점수를 쓴다. 중복도는 항상 코사인이므로 두 값의 스케일이 맞아야
    λ 가 의미를 가진다 — 호출자가 0~1 로 정규화해 넘긴다.
    """
    if lambda_ >= 1.0 or len(candidates) <= 1:
        return candidates[:top_k]

    def _relevance(hit: SearchHit) -> float:
        if relevance is None:
            return hit.score
        return relevance.get(hit.index, hit.score)

    vectors = {hit.index: store.reconstruct(hit.index) for hit in candidates}
    selected: list[SearchHit] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_hit = None
        best_value = -np.inf
        for hit in remaining:
            if selected:
                # 벡터는 저장 시 L2 정규화되어 있으므로 내적이 곧 코사인 유사도다.
                redundancy = max(
                    float(np.dot(vectors[hit.index], vectors[chosen.index]))
                    for chosen in selected
                )
            else:
                redundancy = 0.0
            value = lambda_ * _relevance(hit) - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_value, best_hit = value, hit
        assert best_hit is not None
        selected.append(best_hit)
        remaining.remove(best_hit)

    return selected


def _apply_token_cap(
    hits: list[SearchHit], *, max_tokens: int, model: str
) -> tuple[list[SearchHit], int, int]:
    """토큰 상한 내로 자른다 (PRD F5.4). 유사도가 높은 것부터 담는다."""
    kept: list[SearchHit] = []
    used = 0
    for hit in hits:
        cost = count_tokens(hit.chunk.text, model)
        if kept and used + cost > max_tokens:
            break
        kept.append(hit)
        used += cost
    return kept, used, len(hits) - len(kept)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, cfg: Config) -> None:
        self._store = store
        self._embedder = embedder
        self._cfg = cfg
        self._bm25: BM25Index | None = None

        if cfg.retrieval_mode == "hybrid" and store.size:
            # 섹션 라벨을 문서에 포함한다. 벡터 쪽에는 넣지 않지만(F2.7 — 임베딩
            # 캐시가 깨진다) BM25 는 공짜이고, "특이사항 있어?" 처럼 라벨로만
            # 표현된 내용을 잡아 준다.
            documents = [
                f"{chunk.section} {chunk.text}".strip() for chunk in store.chunks
            ]
            self._bm25 = BM25Index(documents)

    def _cosine(self, query_vector: np.ndarray, index: int) -> float:
        """BM25 만 찾아낸 청크의 코사인 점수를 되계산한다.

        가짜 점수를 넣지 않기 위해서다 — 저장된 벡터는 이미 정규화되어 있으므로
        정규화한 질의 벡터와 내적하면 그게 곧 코사인 유사도다. API 호출은 없다.
        """
        norm = float(np.linalg.norm(query_vector)) or 1.0
        return float(np.dot(self._store.reconstruct(index), query_vector / norm))

    def retrieve(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        cfg = self._cfg
        top_k = top_k or cfg.top_k
        query = expand_query(question, history)

        # 질문당 임베딩 호출은 여기 1회뿐이다.
        query_vector = self._embedder.embed_query(query)
        pool = min(top_k * _CANDIDATE_MULTIPLIER, max(self._store.size, 1))
        raw = self._store.search(query_vector, pool)
        best_score = raw[0].score if raw else 0.0

        above = [hit for hit in raw if hit.score >= cfg.similarity_threshold]
        dropped_threshold = len(raw) - len(above)

        candidates, fused, lexical_only = self._merge_lexical(
            query, above, query_vector, top_k=top_k
        )

        if not candidates:
            logger.info(
                "후보 0건 — LLM 호출 없이 반환 (벡터 최고 %.3f, 임계값 %.2f)",
                best_score,
                cfg.similarity_threshold,
            )
            return RetrievalResult(
                hits=(),
                query=query,
                context_tokens=0,
                dropped_by_threshold=dropped_threshold,
                best_score=best_score,
                mode=cfg.retrieval_mode,
            )

        selected = _mmr_select(
            self._store,
            candidates,
            top_k=top_k,
            lambda_=cfg.mmr_lambda,
            relevance=fused,
        )
        # MMR 이 순서를 흔들어 놓으므로 컨텍스트 순서를 다시 세운다. 하이브리드에서는
        # 융합 점수 순이어야 어휘 신호가 컨텍스트 순서까지 살아남는다.
        selected.sort(
            key=lambda hit: (fused or {}).get(hit.index, hit.score), reverse=True
        )

        kept, used_tokens, dropped_cap = _apply_token_cap(
            selected, max_tokens=cfg.max_context_tokens, model=cfg.llm_model
        )
        kept_indices = {hit.index for hit in kept}

        logger.info(
            "[%s] 벡터 %d건(임계값 통과 %d) + 어휘 추가 %d건 -> 선택 %d건 "
            "(%d tokens, 벡터 최고 %.3f)",
            cfg.retrieval_mode,
            len(raw),
            len(above),
            len(lexical_only),
            len(kept),
            used_tokens,
            best_score,
        )
        return RetrievalResult(
            hits=tuple(kept),
            query=query,
            context_tokens=used_tokens,
            dropped_by_threshold=dropped_threshold,
            dropped_by_token_cap=dropped_cap,
            best_score=best_score,
            mode=cfg.retrieval_mode,
            lexical_only=tuple(
                self._store.chunks[i].block_id
                for i in lexical_only
                if i in kept_indices
            ),
        )

    def _merge_lexical(
        self,
        query: str,
        vector_hits: list[SearchHit],
        query_vector: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[list[SearchHit], dict[int, float] | None, list[int]]:
        """BM25 결과를 벡터 결과와 RRF 로 융합한다.

        반환: (후보 목록, 0~1 로 정규화한 융합 점수, BM25 만 찾아낸 인덱스 목록).
        `vector` 모드거나 BM25 가 아무것도 못 찾으면 융합 점수는 None 이다
        (그때는 MMR 이 코사인 점수를 그대로 쓴다).

        **문턱은 절대 점수가 아니라 그 질의 내 상대 비율이다.** 절대 점수로는
        가를 수 없다는 것을 실측으로 확인했고(모듈 docstring), 상대 비율로는
        깔끔하게 갈렸다 — 잡음 히트는 최고 점수의 0.06~0.49, 정상 히트는 0.52~1.00.
        "환율" 한 단어만 겹친 462자 뉴스 블록이 컨텍스트를 잡아먹는 것을 막는다.
        """
        if self._bm25 is None:
            return vector_hits, None, []

        ranked = self._bm25.search(query)
        if not ranked:
            return vector_hits, None, []

        floor = ranked[0].score * self._cfg.bm25_min_ratio
        lexical = [hit for hit in ranked if hit.score >= floor][:top_k]
        if not lexical:
            return vector_hits, None, []

        by_index = {hit.index: hit for hit in vector_hits}
        lexical_only: list[int] = []

        for hit in lexical:
            if hit.index in by_index:
                continue
            # 임계값을 못 넘었지만 어휘로 잡힌 청크. 점수는 되계산해 진짜 값을 넣는다.
            by_index[hit.index] = SearchHit(
                chunk=self._store.chunks[hit.index],
                score=self._cosine(query_vector, hit.index),
                index=hit.index,
            )
            lexical_only.append(hit.index)

        fused = _rrf_fuse(
            [hit.index for hit in vector_hits],
            [hit.index for hit in lexical],
            rrf_k=self._cfg.rrf_k,
        )
        # MMR 의 중복도(코사인, 0~1)와 스케일을 맞춘다.
        largest = max(fused.values())
        if largest > 0:
            fused = {index: value / largest for index, value in fused.items()}

        candidates = sorted(
            by_index.values(), key=lambda hit: fused.get(hit.index, 0.0), reverse=True
        )
        if lexical_only:
            logger.debug(
                "어휘 검색이 추가한 청크: %s",
                [self._store.chunks[i].block_id for i in lexical_only],
            )
        return candidates, fused, lexical_only
