"""신선도 관리: TTL 판정과 재수집 (PRD F4).

환율은 분 단위로 바뀐다. **낡은 값을 조용히 최신인 척 답하는 것이 이 제품에서 가장
위험한 실패 모드다**(PRD R3). 그래서 질문 시점마다 인덱스의 나이를 확인하고,
TTL 을 넘겼으면 재수집한 뒤에 답한다.

이 모듈은 **수집 계층에 속한다.** `retriever.py` / `chat.py` 는 크롤링을 전혀 알지
못하고, 호출자가 여기서 얻은 최신 store 를 질의 계층에 넘긴다(PRD 아키텍처 분리).

`force=True` 는 **TTL 판정만** 건너뛴다. 본문 해시가 같으면 여전히 재임베딩하지
않는다(F4.5) — 강제 새로고침이 임베딩 비용을 유발해서는 안 된다.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from config import Config

from .ingest import ingest_url, url_hash
from .vectorstore import VectorStore, VectorStoreError, store_dir
from .web_loader import WebLoadError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Freshness:
    crawled_at: str  # ISO8601 UTC. 인덱스가 없으면 ""
    age_minutes: float  # 인덱스가 없거나 시각을 못 읽으면 inf
    ttl_minutes: int
    is_expired: bool

    @property
    def has_index(self) -> bool:
        return bool(self.crawled_at)


@dataclass(frozen=True)
class RefreshOutcome:
    freshness: Freshness
    crawled: bool  # 실제로 페이지를 다시 읽었는가
    reembedded: bool  # 임베딩까지 다시 했는가 (본문이 바뀐 경우)
    stale: bool  # 수집 실패로 낡은 데이터를 쓰고 있는가 (PRD F4.3)
    warning: str = ""  # 사용자에게 보여줄 경고. 없으면 ""


def describe_age(age_minutes: float) -> str:
    """사람이 읽는 경과 시간. F4.3 의 "N분 전 기준" 경고에 쓴다."""
    if age_minutes == float("inf"):
        return "시각 불명"
    minutes = int(age_minutes)
    if minutes < 1:
        return "1분 이내"
    if minutes < 60:
        return f"{minutes}분 전"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}시간 {minutes}분 전" if minutes else f"{hours}시간 전"
    days, hours = divmod(hours, 24)
    return f"{days}일 {hours}시간 전" if hours else f"{days}일 전"


def parse_crawled_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        logger.warning("crawled_at 을 해석할 수 없습니다: %r", value)
        return None
    # 예전 인덱스에 tz 정보가 없을 수 있다. UTC 로 간주한다.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def evaluate(
    store: VectorStore | None, ttl_minutes: int, *, now: datetime | None = None
) -> Freshness:
    """인덱스의 나이와 만료 여부를 계산한다. 네트워크를 쓰지 않는다."""
    now = now or datetime.now(timezone.utc)
    crawled_at = str(store.meta.get("crawled_at", "")) if store is not None else ""
    parsed = parse_crawled_at(crawled_at)

    if store is None or parsed is None:
        return Freshness(
            crawled_at=crawled_at,
            age_minutes=float("inf"),
            ttl_minutes=ttl_minutes,
            is_expired=True,
        )

    age = max(0.0, (now - parsed).total_seconds() / 60.0)
    # ttl 0 은 "질문마다 항상 재수집" 이라는 뜻이다.
    return Freshness(
        crawled_at=crawled_at,
        age_minutes=age,
        ttl_minutes=ttl_minutes,
        is_expired=ttl_minutes <= 0 or age > ttl_minutes,
    )


class FreshIndex:
    """인덱스를 들고 있으면서 TTL 에 따라 스스로 갱신하는 홀더.

    `store` 는 재수집할 때마다 **새 객체로 교체된다.** 질의 계층이 예전 store 를
    붙들고 있으면 낡은 값을 계속 답하게 되므로, 호출자는 매 질문 전에
    `ensure_fresh()` 를 부르고 `store` 를 다시 읽어야 한다.
    """

    def __init__(self, cfg: Config, *, store: VectorStore | None = None) -> None:
        self._cfg = cfg
        self._store = store if store is not None else self._load_local(cfg)
        # 온디맨드 갱신과 백그라운드 갱신이 동시에 대상 페이지를 긁지 않도록 한다
        # (PRD 비기능 "예의": 과도한 요청 금지).
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _load_local(cfg: Config) -> VectorStore | None:
        path = store_dir(cfg.vectorstore_dir, url_hash(cfg.source_url))
        try:
            return VectorStore.load(path, embedding_model=cfg.embedding_model)
        except VectorStoreError as exc:
            logger.info("사용할 수 있는 기존 인덱스가 없습니다: %s", exc)
            return None

    @property
    def store(self) -> VectorStore | None:
        return self._store

    def freshness(self) -> Freshness:
        return evaluate(self._store, self._cfg.crawl_ttl_minutes)

    def ensure_fresh(
        self,
        *,
        force: bool = False,
        allow_network: bool = True,
        offthread: bool = False,
    ) -> RefreshOutcome:
        """필요하면 재수집한다 (PRD F4.1, F4.4).

        `allow_network=False` 는 네트워크를 쓰지 않고 판정만 한다 — 낡았으면
        경고를 붙여 돌려준다(오프라인/디버깅용).

        `offthread=True` 는 **Streamlit 에서 반드시 필요하다.** Streamlit 스크립트
        스레드에는 asyncio 루프가 돌고 있어 `sync_playwright()` 가 거부되기 때문이다.
        새 스레드에는 루프가 없으므로 그쪽으로 밀어낸다.
        """
        # 배포 환경(ENABLE_CRAWL=false)에는 헤드리스 브라우저가 없다. 이 스위치가
        # 꺼져 있으면 수집을 아예 시도하지 않으므로 Playwright 를 import 하지도
        # 않는다 — 앱은 미리 만들어 둔 인덱스를 읽기만 한다.
        if not self._cfg.enable_crawl:
            allow_network = False

        if offthread and allow_network:
            return self._ensure_fresh_offthread(
                force=force, allow_network=allow_network
            )

        with self._lock:
            current = self.freshness()

            if self._store is not None and not force and not current.is_expired:
                return RefreshOutcome(
                    freshness=current, crawled=False, reembedded=False, stale=False
                )

            if not allow_network:
                if self._store is None:
                    raise VectorStoreError(
                        "인덱스가 없습니다. `python index_url.py` 로 먼저 만드세요."
                    )
                return RefreshOutcome(
                    freshness=current,
                    crawled=False,
                    reembedded=False,
                    stale=current.is_expired,
                    warning=self._stale_message(
                        current,
                        "재수집을 건너뛰었습니다"
                        if self._cfg.enable_crawl
                        else "이 환경에서는 재수집하지 않습니다(ENABLE_CRAWL=false)",
                    ),
                )

            reason = "강제 새로고침" if force else f"TTL {current.ttl_minutes}분 경과"
            logger.info("재수집 (%s, 현재 나이 %s)", reason, describe_age(current.age_minutes))

            try:
                # rebuild=False: 본문이 그대로면 재임베딩하지 않는다 (F4.5).
                result = ingest_url(self._cfg)
            except WebLoadError as exc:
                # ingest_url 은 쓸 수 있는 인덱스가 있으면 스스로 폴백한다.
                # 여기까지 예외가 왔다면 인덱스 자체가 없는 것이다.
                logger.error("수집 실패, 폴백할 인덱스도 없습니다: %s", exc)
                raise

            self._store = result.store
            updated = self.freshness()

            if result.stale:
                return RefreshOutcome(
                    freshness=updated,
                    crawled=True,
                    reembedded=False,
                    stale=True,
                    warning=self._stale_message(
                        updated, f"재수집에 실패했습니다 ({result.warnings[0]})"
                    ),
                )

            return RefreshOutcome(
                freshness=updated,
                crawled=True,
                reembedded=not result.from_cache,
                stale=False,
            )

    def _ensure_fresh_offthread(self, **kwargs) -> RefreshOutcome:
        """별도 스레드에서 `ensure_fresh` 를 돌리고 결과·예외를 그대로 전달한다."""
        box: dict[str, object] = {}

        def target() -> None:
            try:
                box["result"] = self.ensure_fresh(**kwargs)
            except BaseException as exc:  # 예외를 삼키면 호출자가 영원히 기다린다
                box["error"] = exc

        thread = threading.Thread(target=target, name="ensure-fresh", daemon=True)
        thread.start()
        thread.join()

        error = box.get("error")
        if error is not None:
            raise error  # type: ignore[misc]
        result = box.get("result")
        assert isinstance(result, RefreshOutcome)
        return result

    @staticmethod
    def _stale_message(freshness: Freshness, reason: str) -> str:
        """PRD F4.3 — 낡음을 숨기지 않고 나이와 함께 밝힌다."""
        return (
            f"{reason}. {describe_age(freshness.age_minutes)} 수집된 데이터로 "
            f"답하고 있습니다."
        )

    # ------------------------------------------------------------------ #
    # 백그라운드 주기 갱신 (PRD F4.6)
    #
    # APScheduler 대신 데몬 스레드를 쓴다 — 주기 실행 하나뿐이라 의존성을
    # 늘릴 이유가 없다. Event.wait() 로 자므로 정지 요청에 즉시 반응한다.
    # ------------------------------------------------------------------ #

    def start_background(self, *, interval_minutes: int | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        interval = max(1, interval_minutes or self._cfg.crawl_ttl_minutes or 1)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(interval,), name="freshness", daemon=True
        )
        self._thread.start()
        logger.info("백그라운드 갱신 시작 (%d분 주기)", interval)

    def stop_background(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        self._thread = None

    def _loop(self, interval_minutes: int) -> None:
        while not self._stop.wait(interval_minutes * 60):
            try:
                outcome = self.ensure_fresh()
                if outcome.crawled:
                    logger.info(
                        "백그라운드 갱신 완료 (재임베딩 %s)",
                        "함" if outcome.reembedded else "안 함",
                    )
            except Exception as exc:  # 스레드가 죽으면 갱신이 조용히 멈춘다
                logger.warning("백그라운드 갱신 실패, 다음 주기에 재시도: %s", exc)
