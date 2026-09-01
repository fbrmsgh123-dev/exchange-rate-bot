"""M5 Streamlit UI (PRD F6.8, F7).

    streamlit run app.py

파이프라인 모듈은 Streamlit 을 알지 못한다 — 이 파일만 알고 있다. CLI(`ask.py`)와
동일한 `FreshIndex` + `ChatEngine` 을 쓰고, 여기서는 상태 보관과 표시만 담당한다.

**Streamlit 고유 함정**: 스크립트 스레드에는 asyncio 루프가 돌고 있어
`sync_playwright()` 가 거부된다. 그래서 재수집은 반드시 `offthread=True` 로 호출한다.
"""

from __future__ import annotations

import dataclasses
import logging
import os

import streamlit as st

from config import Config, ConfigError, get_config
from src.chat import ChatEngine
from src.embedder import Embedder
from src.freshness import FreshIndex, describe_age
from src.utils import setup_logging
from src.vectorstore import VectorStoreError
from src.web_loader import WebLoadError

DISCLAIMER = (
    "표시 값은 원본 페이지에 게시된 내용이며 실거래 환율과 다를 수 있습니다. "
    "본 서비스는 금융 조언을 제공하지 않습니다."
)

st.set_page_config(page_title="환율 페이지 Q&A", page_icon="💱", layout="centered")


def _bridge_secrets() -> None:
    """`st.secrets` 를 환경변수로 옮긴다.

    `config.py` 는 `os.environ` 만 읽는다(다른 모듈이 Streamlit 에 의존하지 않게
    하려는 설계). Streamlit Cloud 에는 `.env` 가 없으므로 여기서 다리를 놓는다.
    이미 환경변수가 있으면 건드리지 않는다 — 로컬 `.env` 가 우선이다.
    """
    try:
        secrets = st.secrets
    except Exception:  # secrets.toml 이 없으면 접근 자체가 실패할 수 있다
        return
    for key, value in secrets.items():
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value


_bridge_secrets()


# --------------------------------------------------------------------------- #
# 상태
# --------------------------------------------------------------------------- #


def _init_state(cfg: Config) -> None:
    st.session_state.setdefault("cfg", cfg)
    st.session_state.setdefault("messages", [])  # 화면 표시용
    st.session_state.setdefault("history", [])  # 엔진 전달용 (질문, 답변)
    st.session_state.setdefault("tokens", 0)
    st.session_state.setdefault("llm_calls", 0)
    st.session_state.setdefault("skipped_calls", 0)


def _build_index(cfg: Config, *, force: bool) -> tuple[FreshIndex, str]:
    """인덱스를 준비한다. 필요하면 수집한다 (PRD F4.1)."""
    index = FreshIndex(cfg)
    freshness = index.freshness()

    if force or freshness.is_expired:
        label = (
            "페이지를 수집하고 인덱싱하는 중… (최초 실행은 30초 정도 걸립니다)"
            if not freshness.has_index
            else "페이지를 다시 읽는 중…"
        )
        with st.spinner(label):
            # offthread: Streamlit 스레드에서는 Playwright 동기 API 가 거부된다.
            outcome = index.ensure_fresh(force=force, offthread=True)
        return index, outcome.warning

    return index, ""


def _reset_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["history"] = []
    st.session_state["tokens"] = 0
    st.session_state["llm_calls"] = 0
    st.session_state["skipped_calls"] = 0


# --------------------------------------------------------------------------- #
# 표시
# --------------------------------------------------------------------------- #


def _render_sources(entry: dict) -> None:
    """근거 스니펫을 접힘 UI 로 (PRD F6.8)."""
    hits = entry.get("sources") or []
    if not hits:
        if entry.get("empty_retrieval"):
            st.caption(
                f"검색 결과 0건 (최고 유사도 {entry.get('best_score', 0):.3f} < 임계값) "
                f"— LLM 을 호출하지 않았습니다. 토큰 0"
            )
        return

    with st.expander(f"근거 {len(hits)}건 · 출처 보기"):
        for hit in hits:
            section = f"**{hit['section']}** · " if hit["section"] else ""
            st.markdown(f"{section}유사도 `{hit['score']:.3f}` · 블록 `#{hit['block_id']}`")
            st.caption(hit["text"])
        if entry.get("source_url"):
            st.markdown(f"[원본 페이지 열기]({entry['source_url']})")

    if entry.get("llm_called"):
        st.caption(
            f"토큰: 프롬프트 {entry['prompt_tokens']} + 응답 "
            f"{entry['completion_tokens']} = {entry['total_tokens']}"
        )


def _render_history() -> None:
    for entry in st.session_state["messages"]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                if entry.get("stale_warning"):
                    st.warning(entry["stale_warning"])
                _render_sources(entry)


# --------------------------------------------------------------------------- #
# 메인
# --------------------------------------------------------------------------- #


def main() -> None:
    setup_logging(logging.WARNING)

    try:
        base_cfg = get_config()
    except ConfigError as exc:
        st.error(f"설정 오류: {exc}")
        st.stop()

    _init_state(base_cfg)
    cfg: Config = st.session_state["cfg"]

    # ---- 사이드바 ---------------------------------------------------------- #
    with st.sidebar:
        st.header("설정")

        url_input = st.text_input("소스 URL", value=cfg.source_url, help="PRD F7.3")
        if url_input.strip() and url_input.strip() != cfg.source_url:
            if st.button("URL 적용", width="stretch"):
                # 소스가 바뀌면 이전 대화는 다른 문서에 대한 것이므로 함께 비운다.
                st.session_state["cfg"] = dataclasses.replace(
                    cfg, source_url=url_input.strip()
                )
                _reset_conversation()
                st.session_state.pop("index", None)
                st.rerun()

        top_k = st.slider(
            "Top-K (검색 청크 수)", min_value=3, max_value=10, value=cfg.top_k,
            help="PRD F5.2 — 늘리면 근거가 많아지고 토큰도 늘어납니다",
        )
        st.caption(
            f"검색 {cfg.retrieval_mode} · 임계값 {cfg.similarity_threshold} · "
            f"컨텍스트 상한 {cfg.max_context_tokens} tokens · "
            f"이력 최근 {cfg.max_history_turns}턴"
        )

        st.divider()
        refresh_clicked = (
            st.button("페이지 새로고침", width="stretch", help="PRD F4.4")
            if cfg.enable_crawl
            else False
        )
        if not cfg.enable_crawl:
            st.caption(
                "읽기 전용 배포입니다. 인덱스는 외부 작업(GitHub Actions 등)이 "
                "갱신합니다."
            )
        reset_clicked = st.button("대화 초기화", width="stretch", help="PRD F7.3")

        st.divider()
        st.metric("누적 토큰", f"{st.session_state['tokens']:,}")
        st.caption(
            f"LLM 호출 {st.session_state['llm_calls']}회 · "
            f"미호출(근거 0건) {st.session_state['skipped_calls']}회"
        )
        st.caption(
            f"모델: `{cfg.llm_model}` / `{cfg.embedding_model}`. "
            "단가가 확인되지 않은 모델이라 비용은 표시하지 않습니다."
        )

    if reset_clicked:
        _reset_conversation()
        st.rerun()

    # ---- 인덱스 준비 ------------------------------------------------------- #
    if refresh_clicked:
        st.session_state.pop("index", None)

    if "index" not in st.session_state:
        try:
            index, warning = _build_index(cfg, force=refresh_clicked)
        except WebLoadError as exc:
            st.error(f"페이지를 수집할 수 없습니다: {exc}")
            st.info("잠시 후 새로고침하거나, 사이드바에서 소스 URL을 확인하세요.")
            st.stop()
        except VectorStoreError as exc:
            st.error(f"인덱스를 찾을 수 없습니다: {exc}")
            if not cfg.enable_crawl:
                st.info(
                    "이 배포는 미리 만들어 둔 인덱스를 읽기만 합니다"
                    "(`ENABLE_CRAWL=false`). `vectorstore/` 가 배포본에 포함되어야 "
                    "합니다 — 자세한 절차는 DEPLOY.md 참조."
                )
            st.stop()
        st.session_state["index"] = index
        st.session_state["startup_warning"] = warning

    index: FreshIndex = st.session_state["index"]
    store = index.store
    if store is None:
        st.error("인덱스를 준비할 수 없습니다.")
        st.stop()

    if "engine" not in st.session_state:
        st.session_state["engine"] = ChatEngine(
            store,
            Embedder(api_key=cfg.openai_api_key, model=cfg.embedding_model),
            cfg,
        )
    engine: ChatEngine = st.session_state["engine"]
    engine.set_store(store)

    # ---- 헤더 -------------------------------------------------------------- #
    st.title("💱 환율 페이지 Q&A")
    freshness = index.freshness()
    left, right = st.columns(2)
    left.metric("데이터 기준 시각", engine.data_timestamp())
    right.metric(
        "인덱스 나이",
        describe_age(freshness.age_minutes),
        help=f"TTL {cfg.crawl_ttl_minutes}분 경과 시 질문할 때 자동 재수집합니다",
    )
    st.caption(f"소스: {store.meta.get('source_url', cfg.source_url)} · 청크 {store.size}개")

    if st.session_state.get("startup_warning"):
        st.warning(st.session_state["startup_warning"])
    st.info(DISCLAIMER, icon="ℹ️")

    _render_history()

    # ---- 질문 -------------------------------------------------------------- #
    question = st.chat_input("예: 달러 환율 얼마야?")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 질문 시점 신선도 확인 (PRD F4.1)
    warning = ""
    try:
        outcome = index.ensure_fresh(offthread=True)
        if outcome.crawled:
            st.toast(
                f"페이지를 다시 읽었습니다 (재임베딩 "
                f"{'함' if outcome.reembedded else '안 함'})"
            )
        warning = outcome.warning
        engine.set_store(index.store)
    except WebLoadError as exc:
        warning = f"재수집에 실패해 이전 데이터로 답합니다: {exc}"

    with st.chat_message("assistant"):
        if warning:
            st.warning(warning)
        placeholder = st.empty()
        pieces: list[str] = []

        def on_token(piece: str) -> None:
            pieces.append(piece)
            placeholder.markdown("".join(pieces))

        try:
            answer = engine.ask(
                question,
                st.session_state["history"],
                stream=True,
                on_token=on_token,
                stale_warning=warning,
                top_k=top_k,
            )
        except Exception as exc:
            placeholder.empty()
            st.error(f"답변 생성에 실패했습니다: {exc}")
            st.session_state["messages"].pop()  # 답 없는 질문은 이력에 남기지 않는다
            return

        placeholder.markdown(answer.text)

        entry = {
            "role": "assistant",
            "content": answer.text,
            "stale_warning": answer.stale_warning,
            "llm_called": answer.llm_called,
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "total_tokens": answer.total_tokens,
            "empty_retrieval": answer.retrieval.is_empty,
            "best_score": answer.retrieval.best_score,
            "source_url": str(store.meta.get("source_url", cfg.source_url)),
            "sources": [
                {
                    "score": hit.score,
                    "block_id": hit.chunk.block_id,
                    "section": hit.chunk.section,
                    "text": hit.chunk.text,
                }
                for hit in answer.retrieval.hits
            ],
        }
        _render_sources(entry)

    st.session_state["messages"].append(entry)
    st.session_state["history"].append((question, answer.text))
    st.session_state["tokens"] += answer.total_tokens
    if answer.llm_called:
        st.session_state["llm_calls"] += 1
    else:
        st.session_state["skipped_calls"] += 1

    # 사이드바(누적 토큰·호출 횟수)는 이 지점보다 **먼저** 그려졌으므로, 다시 실행하지
    # 않으면 방금 쓴 토큰이 반영되지 않는다. 이력은 session_state 에서 다시 그려진다.
    st.rerun()


main()
