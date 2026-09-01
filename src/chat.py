"""프롬프트 조립 + LLM 호출 (PRD F6, F7).

**두 가지 불변 조건을 이 모듈이 지킨다.**

1. 검색 결과가 0건이면 LLM 을 호출하지 않는다 (PRD F5.5). 즉시 정해진 문구를
   돌려주어 그 질문의 LLM 비용을 0으로 만든다.
2. 컨텍스트는 검색된 청크뿐이다. 페이지 전문을 넣지 않는다.

대화 이력은 최근 `MAX_HISTORY_TURNS` 턴만 넣는다 (F7.2). 대화가 길어져도 프롬프트
크기가 상수로 유지되어야 한다.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from openai import BadRequestError, OpenAI

from config import Config

from .embedder import Embedder
from .retriever import RetrievalResult, Retriever
from .utils import count_tokens
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)

# 근거가 없을 때의 응답. PRD F5.5 / F6.3 이 문구를 규정한다.
NO_CONTEXT_MESSAGE = "제공된 페이지에서 해당 내용을 찾을 수 없습니다."

_SYSTEM_PROMPT = """당신은 아래 <컨텍스트>에 주어진 웹페이지 발췌 내용에 대해서만 답변하는 어시스턴트입니다.
소스: {source_url}
데이터 기준 시각: {data_timestamp}

규칙:
1. <컨텍스트>에 있는 내용만 근거로 답변합니다. 사전 지식으로 환율 값을 채워 넣지 않습니다.
2. 컨텍스트에 없는 내용은 추측하지 말고 "{no_context}"라고 답합니다.
3. 숫자는 컨텍스트에 적힌 값을 그대로 인용합니다. 임의로 계산·환산·반올림하지 않습니다.
   계산을 요청받은 경우에만 계산식을 함께 밝힙니다.
4. 답변 끝에 데이터 기준 시각을 [기준: {data_timestamp}] 형식으로 표기합니다.
5. 환율 예측이나 투자 판단은 제공하지 않습니다. 요청받으면 본 서비스가 페이지에 게시된
   값을 전달할 뿐 금융 조언을 제공하지 않는다고 밝힙니다.
6. 사용자의 질문 언어와 동일한 언어로 답변합니다.

<컨텍스트>
{context}
</컨텍스트>"""

# 페이지가 스스로 표기하는 갱신 시각 (PRD F4.2). 우리 크롤링 시각보다 정확한 근거다.
_PAGE_TIMESTAMP_PATTERNS = (
    re.compile(r"마지막\s*갱신\s*·?\s*(\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2})"),
    re.compile(r"갱신\s*(\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2})"),
)


@dataclass(frozen=True)
class Answer:
    text: str
    retrieval: RetrievalResult
    data_timestamp: str
    llm_called: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stale_warning: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def page_timestamp(store: VectorStore) -> str:
    """데이터 기준 시각을 정한다 (PRD F4.2).

    페이지가 "마지막 갱신 · 09/01 15:05" 처럼 스스로 밝힌 값이 있으면 그것을 쓴다.
    우리가 언제 크롤링했는지보다 **데이터가 언제 만들어졌는지**가 사용자에게 맞는
    근거이기 때문이다. 없으면 `crawled_at` 으로 대체한다.
    """
    for chunk in store.chunks:
        for pattern in _PAGE_TIMESTAMP_PATTERNS:
            found = pattern.search(chunk.text)
            if found:
                return found.group(1)

    crawled = str(store.meta.get("crawled_at", ""))
    if not crawled:
        return "알 수 없음"
    try:
        # 저장은 UTC ISO8601. 보여줄 때는 로컬 시간으로 바꾼다.
        return (
            datetime.fromisoformat(crawled).astimezone().strftime("%m/%d %H:%M")
            + " (수집 시각)"
        )
    except ValueError:
        return crawled


def build_context(result: RetrievalResult) -> str:
    """검색된 청크만 컨텍스트로 만든다.

    블록 번호와 섹션을 함께 넣어 모델이 출처를 밝힐 수 있게 한다. 수집 시각을
    청크 텍스트에 박아 넣지 않고 여기서 조립하는 이유는 PRD F2.7 참조.
    """
    lines = []
    for hit in result.hits:
        chunk = hit.chunk
        label = f"#{chunk.block_id}"
        if chunk.section:
            label += f" · {chunk.section}"
        lines.append(f"[{label}]\n{chunk.text}")
    return "\n\n".join(lines)


def build_messages(
    question: str,
    result: RetrievalResult,
    *,
    source_url: str,
    data_timestamp: str,
    history: Sequence[tuple[str, str]] = (),
    max_history_turns: int = 4,
) -> list[dict]:
    system = _SYSTEM_PROMPT.format(
        source_url=source_url,
        data_timestamp=data_timestamp,
        no_context=NO_CONTEXT_MESSAGE,
        context=build_context(result),
    )
    messages: list[dict] = [{"role": "system", "content": system}]

    # 최근 N턴만. 이력이 길어져도 프롬프트가 커지지 않아야 한다 (PRD F7.2).
    for past_question, past_answer in list(history)[-max_history_turns:]:
        messages.append({"role": "user", "content": past_question})
        messages.append({"role": "assistant", "content": past_answer})

    messages.append({"role": "user", "content": question})
    return messages


class ChatEngine:
    def __init__(self, store: VectorStore, embedder: Embedder, cfg: Config) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self.set_store(store)

    def set_store(self, store: VectorStore) -> None:
        """인덱스를 교체한다 (PRD F4.1).

        재수집하면 `FreshIndex` 가 **새 store 객체**를 만든다. 여기서 갈아끼우지
        않으면 엔진이 예전 인덱스를 붙들고 낡은 환율을 계속 답한다.
        """
        if getattr(self, "_store", None) is store:
            return
        self._store = store
        self._retriever = Retriever(store, self._embedder, self._cfg)
        self._source_url = str(store.meta.get("source_url", self._cfg.source_url))

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    def data_timestamp(self) -> str:
        return page_timestamp(self._store)

    def ask(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
        *,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
        stale_warning: str = "",
        top_k: int | None = None,
    ) -> Answer:
        result = self._retriever.retrieve(question, history, top_k=top_k)
        timestamp = self.data_timestamp()

        # PRD F5.5 — 근거가 없으면 LLM 을 호출하지 않는다. 이 질문의 LLM 비용은 0.
        if result.is_empty:
            text = NO_CONTEXT_MESSAGE
            if on_token is not None:
                on_token(text)
            answer = Answer(
                text=text,
                retrieval=result,
                data_timestamp=timestamp,
                llm_called=False,
                stale_warning=stale_warning,
            )
            self._log_query(question, answer)
            return answer

        messages = build_messages(
            question,
            result,
            source_url=self._source_url,
            data_timestamp=timestamp,
            history=history,
            max_history_turns=self._cfg.max_history_turns,
        )
        prompt_tokens = sum(
            count_tokens(str(m["content"]), self._cfg.llm_model) for m in messages
        )

        text, completion_tokens = self._complete(
            messages, stream=stream, on_token=on_token
        )

        answer = Answer(
            text=text,
            retrieval=result,
            data_timestamp=timestamp,
            llm_called=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            stale_warning=stale_warning,
        )
        self._log_query(question, answer)
        return answer

    def _log_query(self, question: str, answer: Answer) -> None:
        """질문·검색 청크 ID·토큰 사용량을 로컬 로그에 남긴다 (PRD 비기능 로깅).

        청크 **원문은 기록하지 않는다** — ID 만으로 어떤 근거가 쓰였는지 추적된다.
        로깅 실패가 답변을 막아서는 안 되므로 예외는 삼킨다.
        """
        path = self._cfg.query_log_path
        if path is None:
            return
        record = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question,
            "search_query": answer.retrieval.query,
            "blocks": [hit.chunk.block_id for hit in answer.retrieval.hits],
            "scores": [round(hit.score, 4) for hit in answer.retrieval.hits],
            "llm_called": answer.llm_called,
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "data_timestamp": answer.data_timestamp,
            "stale": bool(answer.stale_warning),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("질의 로그를 쓸 수 없습니다 (%s): %s", path, exc)

    # ------------------------------------------------------------------ #
    # LLM 호출
    #
    # 모델(`gpt-5.6-sol`)의 스트리밍 지원 여부·파라미터 허용 범위가 확인되지
    # 않았으므로(PRD Q4) `temperature` 나 `max_tokens` 를 보내지 않는다. 모델
    # 특성을 가정한 코드를 넣지 말 것 — 환경변수로 교체 가능해야 한다.
    # ------------------------------------------------------------------ #

    def _complete(
        self,
        messages: list[dict],
        *,
        stream: bool,
        on_token: Callable[[str], None] | None,
    ) -> tuple[str, int]:
        if stream:
            try:
                return self._stream_call(messages, on_token)
            except BadRequestError as exc:
                # 스트리밍을 거부하는 모델일 수 있다. 400 은 그 신호로 본다.
                logger.warning("스트리밍이 거부되어 일반 호출로 대체합니다: %s", exc)
        return self._plain_call(messages, on_token)

    def _stream_call(
        self, messages: list[dict], on_token: Callable[[str], None] | None
    ) -> tuple[str, int]:
        pieces: list[str] = []
        completion_tokens = 0
        response = self._client.chat.completions.create(
            model=self._cfg.llm_model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in response:
            if event.usage is not None:
                completion_tokens = event.usage.completion_tokens
            if not event.choices:
                continue
            piece = event.choices[0].delta.content
            if piece:
                pieces.append(piece)
                if on_token is not None:
                    on_token(piece)
        text = "".join(pieces)
        if not completion_tokens:
            completion_tokens = count_tokens(text, self._cfg.llm_model)
        return text, completion_tokens

    def _plain_call(
        self, messages: list[dict], on_token: Callable[[str], None] | None
    ) -> tuple[str, int]:
        response = self._client.chat.completions.create(
            model=self._cfg.llm_model, messages=messages
        )
        text = response.choices[0].message.content or ""
        if on_token is not None:
            on_token(text)
        completion_tokens = (
            response.usage.completion_tokens
            if response.usage is not None
            else count_tokens(text, self._cfg.llm_model)
        )
        return text, completion_tokens
