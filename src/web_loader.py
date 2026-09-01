"""URL -> 렌더링된 페이지 -> 의미 블록 (PRD F1, F2.1/F2.2/F2.5/F2.6).

**두 단계를 일부러 분리했다.**

- `render_html()`  : 네트워크 + 헤드리스 브라우저가 필요한 단계
- `extract_blocks()`: 순수 함수. 저장된 `snapshot.html` 만으로 재현·검증된다.

대상 페이지의 DOM 이 바뀌었을 때(PRD R5) 브라우저를 띄우지 않고 파서만 고쳐
검증할 수 있어야 하기 때문이다. `load_html()` 로 스냅샷을 그대로 다시 파싱할 수 있다.

주의: `sync_playwright()` 는 같은 스레드에 실행 중인 asyncio 루프가 있으면 예외를
던진다. Streamlit(M5)에서 호출할 때는 별도 스레드에서 돌려야 한다.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup, Tag

from .utils import bytes_hash

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 예외 — 메시지는 그대로 사용자에게 보여줄 수 있어야 한다 (PRD §8)
# --------------------------------------------------------------------------- #


class WebLoadError(RuntimeError):
    """페이지를 수집할 수 없을 때의 기반 예외."""


class BrowserUnavailableError(WebLoadError):
    """playwright / chromium 이 준비되지 않았을 때."""


class AuthRequiredError(WebLoadError):
    """인증 페이지로 리다이렉트되었을 때 (PRD F1.5)."""


class AppAsleepError(WebLoadError):
    """Streamlit Community Cloud 앱이 슬립 상태일 때 (PRD R2)."""


class RenderTimeoutError(WebLoadError):
    """제한 시간 안에 렌더링이 끝나지 않았을 때 (PRD F1.3)."""


class EmptyContentError(WebLoadError):
    """렌더링은 됐지만 본문 텍스트가 없을 때 (PRD F1.4)."""


# --------------------------------------------------------------------------- #
# 데이터 모델
# --------------------------------------------------------------------------- #

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

BLOCK_TYPES = (
    "heading",
    "paragraph",
    "list",
    "table_row",
    "metric",
    "code",
)


@dataclass(frozen=True)
class Block:
    """청킹의 입력 단위. `page` 대신 `block_id`/`section` 이 출처가 된다 (PRD F2.3)."""

    block_id: int
    block_type: str
    section: str  # 직전 heading. 없으면 ""
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExtractResult:
    blocks: tuple[Block, ...]
    warnings: tuple[str, ...] = ()
    coverage: float = 1.0  # 블록으로 잡힌 글자 수 / 본문 전체 글자 수 (PRD R5)


@dataclass(frozen=True)
class RenderedPage:
    html: str  # 본문이 들어있는 프레임의 HTML
    final_url: str  # 최상위 프레임의 최종 URL (인증 리다이렉트 판정용)
    frame_url: str  # 본문을 읽어온 프레임의 URL
    text: str  # 그 프레임의 body innerText


@dataclass(frozen=True)
class PageSnapshot:
    url: str
    final_url: str
    crawled_at: str  # ISO8601 UTC
    html: str
    text: str  # 브라우저가 본 body innerText (디버깅용 원본)
    blocks: tuple[Block, ...]
    warnings: tuple[str, ...] = field(default=())
    frame_url: str = ""  # 본문을 읽어온 프레임 (최상위와 다를 수 있다)

    @property
    def content_text(self) -> str:
        """블록을 이어붙인 본문. 해시·저장·dry-run 출력의 기준이다."""
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def content_hash(self) -> str:
        """본문 해시. 값이 그대로면 재임베딩을 건너뛴다 (PRD F3.5, F4.5).

        상대 시각("· 4분 전")은 해시에서 제외한다. 이 문구는 페이지를 볼 때마다
        1분 단위로 바뀌므로, 그대로 해시하면 환율이 하나도 안 변했는데도 매번
        해시가 달라져 재임베딩 생략(F4.5)이 전혀 동작하지 않는다. 블록 텍스트
        자체에는 남겨 둔다 — 답변에 신선도를 알려주는 데 쓸모가 있다.
        """
        stable = _RELATIVE_TIME.sub("", self.content_text)
        return bytes_hash(stable.encode("utf-8"))


# --------------------------------------------------------------------------- #
# 상용구 제거 (PRD F2.5)
# --------------------------------------------------------------------------- #

# 통째로 버릴 요소. Streamlit 의 크롬(toolbar/footer 등)이 대부분이다.
_DROP_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "head",
    "header",
    "footer",
    '[data-testid="stToolbar"]',
    '[data-testid="stDecoration"]',
    '[data-testid="stStatusWidget"]',
    '[data-testid="stHeader"]',
    '[data-testid="stAppDeployButton"]',
    '[data-testid="stMainMenu"]',
    '[data-testid="stBottomBlockContainer"]',
    '[data-testid="manage-app-button"]',
    "#MainMenu",
    # 위젯은 조작 대상이지 본문이 아니다 (PRD §2.2 — 위젯 조작은 v1 범위 밖).
    # 남겨두면 차트 계열 토글 라벨 같은 UI 문구가 그대로 청크에 섞인다.
    '[data-testid="stCheckbox"]',
    '[data-testid="stRadio"]',
    '[data-testid="stButton"]',
    '[data-testid="stSelectbox"]',
    '[data-testid="stSlider"]',
    '[data-testid="stTextInput"]',
    '[data-testid="stDateInput"]',
    '[data-testid="stWidgetLabel"]',
    # 차트의 시각적 해석은 비목표. svg 를 걷어낸 뒤 남는 축·범례 문구만 노이즈가 된다.
    '[data-testid="stPlotlyChart"]',
    '[data-testid="stVegaLiteChart"]',
    '[data-testid="stArrowVegaLiteChart"]',
)

# Streamlit 은 st.* 호출 하나를 stElementContainer 하나로 감싼다. 즉 이것이
# **작성자가 의도한 의미 단위**다. 이 경계에서 묶지 않으면 카드 하나가
# "미국 달러" / "1 USD" / "1,370.60" / "▲ 1.10" 처럼 조각조각 흩어져, 청크 하나만
# 봐서는 어느 통화의 값인지 알 수 없게 된다 (PRD F2.2 가 표에서 막으려던 것과 같은 문제).
_ELEMENT_CONTAINERS = ("stElementContainer",)

# 안에 이런 구조가 있으면 통째로 묶지 않고 각각의 규칙으로 처리한다.
_STRUCTURED_TAGS = ("table", "ul", "ol", "pre", *_HEADING_TAGS)

# 블록 텍스트가 이 패턴과 정확히 일치하면 버린다.
_BOILERPLATE = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^hosted with streamlit$",
        r"^made with streamlit$",
        r"^created by$",
        r"^manage app$",
        r"^deploy$",
        r"^fork$",
        r"^star$",
        r"^rerun$",
        r"^settings$",
        r"^about$",
        r"^print$",
        r"^clear cache$",
        r"^(view )?source( code)?$",
        r"^record a screencast$",
        r"^report a bug.*$",
        r"^get help.*$",
        r"^edit(ing)? in .*$",
    )
)

# 본문 루트 후보. 앞에서부터 먼저 발견된 것을 쓴다.
_CONTENT_SELECTORS = (
    '[data-testid="stAppViewContainer"]',
    '[data-testid="stMain"]',
    '[data-testid="stAppViewBlockContainer"]',
    "section.main",
    "main",
    "body",
)

# 슬립/에러 상태 판별 문구 (PRD §8)
_ASLEEP_MARKERS = (
    "has gone to sleep",
    "get this app back up",
    "app is sleeping",
)
_APP_ERROR_MARKERS = (
    "error running app",
    "oh no.",
    "connection error",
)

_WS = re.compile(r"\s+")
# 숫자가 span 으로 쪼개져 "1,384. 50" 처럼 벌어지는 것을 되붙인다.
_DIGIT_GAP = re.compile(r"(?<=[\d,.]) +(?=[\d,.])")
# 폭 0 문자. 대상 페이지는 줄바꿈 방지용으로 U+2060 을 단어 사이에 넣어
# "미⁠국 달⁠러" 처럼 만든다. 지우지 않으면 "미국"으로 검색해도 매칭되지 않는다.
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\ufeff\u00ad]")

# 이모지로 시작하는 짧은 줄을 소제목으로 본다. 대상 페이지에는 h1~h6 이 하나도
# 없고 "📈 30일 추이" 같은 문구가 st.markdown 으로만 표현되기 때문이다(PRD Q1).
# 통화 카드("🇺🇸 미국 달러 1 USD 1,370.60 ...")가 잘못 잡히지 않도록 소수점 숫자를
# 포함한 줄은 제외한다.
_LEADING_EMOJI = re.compile(
    r"^[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]"
)
_RATE_NUMBER = re.compile(r"\d[\d,]*\.\d")
_HEADING_MAX_CHARS = 30

# "· 4분 전" 처럼 읽는 시점에 따라 달라지는 상대 시각. 해시 계산에서만 제거한다.
_RELATIVE_TIME = re.compile(r"[·,]?\s*\d+\s*(?:초|분|시간|일)\s*전")


def _collapse(text: str) -> str:
    return _WS.sub(" ", _ZERO_WIDTH.sub("", text)).strip()


def _looks_like_heading(text: str) -> bool:
    return (
        len(text) <= _HEADING_MAX_CHARS
        and bool(_LEADING_EMOJI.match(text))
        and not _RATE_NUMBER.search(text)
    )


def _squeeze(text: str) -> str:
    """길이 비교용. 모든 공백을 제거한다."""
    return _WS.sub("", text)


def _numeric_text(tag: Tag) -> str:
    """숫자 셀/지표용 텍스트. 자릿수 사이에 공백이 끼지 않게 한다."""
    return _DIGIT_GAP.sub("", _collapse(tag.get_text(" ", strip=True)))


def _is_boilerplate(text: str) -> bool:
    return any(pattern.match(text) for pattern in _BOILERPLATE)


# --------------------------------------------------------------------------- #
# 블록 추출 (PRD F2.1, F2.2)
# --------------------------------------------------------------------------- #


class _Collector:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.section = ""

    def add(self, block_type: str, text: str) -> None:
        text = text.strip()
        if not text or _is_boilerplate(text):
            return
        self.blocks.append(
            Block(
                block_id=len(self.blocks),
                block_type=block_type,
                section=self.section,
                text=text,
            )
        )


def _metric_block_text(tag: Tag) -> str:
    """st.metric 위젯 -> "라벨: 값 (전일대비 +3.20)".

    증감 방향은 SVG 화살표로만 표시되고 텍스트에는 부호가 없다. `_DROP_SELECTORS`
    가 svg 를 걷어내므로, 방향은 아이콘 컨테이너의 testid(`...-Up`/`-Down`)에서
    복원한다. 이걸 빼먹으면 "+3.20" 과 "-3.20" 이 구분되지 않는다.
    """
    label_tag = tag.select_one('[data-testid="stMetricLabel"]')
    value_tag = tag.select_one('[data-testid="stMetricValue"]')
    delta_tag = tag.select_one('[data-testid="stMetricDelta"]')

    label = _collapse(label_tag.get_text(" ", strip=True)) if label_tag else ""
    value = _numeric_text(value_tag) if value_tag else ""
    if not value:
        return ""

    text = f"{label}: {value}" if label else value

    if delta_tag is not None:
        delta = _numeric_text(delta_tag)
        icon = delta_tag.select_one('[data-testid^="stMetricDeltaIcon-"]')
        sign = ""
        if icon is not None:
            testid = icon.get("data-testid", "")
            if testid.endswith("-Up"):
                sign = "+"
            elif testid.endswith("-Down"):
                sign = "-"
        if delta and not delta.startswith(("+", "-")):
            delta = f"{sign}{delta}"
        if delta:
            text = f"{text} (변동 {delta})"
    return text


def _row_cells(row: Tag) -> list[str]:
    return [_numeric_text(cell) for cell in row.find_all(("th", "td"), recursive=False)]


def _table_headers(table: Tag) -> tuple[list[str], set[int]]:
    """헤더 텍스트와, 본문에서 제외해야 할 행의 id 집합을 돌려준다."""
    head = table.find("thead")
    if head is not None:
        rows = head.find_all("tr")
        if rows:
            return _row_cells(rows[-1]), {id(r) for r in rows}

    # thead 가 없으면 첫 행이 전부 th 일 때만 헤더로 본다.
    first = table.find("tr")
    if first is not None and first.find("td") is None and first.find("th") is not None:
        return _row_cells(first), {id(first)}
    return [], set()


def _table_rows(table: Tag) -> list[str]:
    """표를 **행 단위**로 펼친다 (PRD F2.2).

    통짜 텍스트로 넘기면 행-열 대응이 깨져 숫자 답변이 틀린다. 각 셀에 헤더를
    붙여 행 하나만 봐도 의미가 통하게 만든다.
    """
    headers, skip = _table_headers(table)

    texts: list[str] = []
    for row in table.find_all("tr"):
        if id(row) in skip:
            continue
        cells = _row_cells(row)
        if not any(cells):
            continue
        if headers and len(headers) == len(cells):
            parts = [
                f"{header} {cell}".strip() if header else cell
                for header, cell in zip(headers, cells)
                if cell
            ]
        else:
            parts = [cell for cell in cells if cell]
        if parts:
            texts.append(" | ".join(parts))
    return texts


def _list_block_text(tag: Tag) -> str:
    items = []
    for item in tag.find_all("li", recursive=False):
        text = _collapse(item.get_text(" ", strip=True))
        if text:
            items.append(f"- {text}")
    return "\n".join(items)


def _is_metric(tag: Tag) -> bool:
    return tag.get("data-testid") == "stMetric"


def _has_structured_content(tag: Tag) -> bool:
    """표·목록·제목·지표가 들어있으면 통째로 묶지 않고 개별 규칙으로 처리한다."""
    return (
        tag.find(_STRUCTURED_TAGS) is not None
        or tag.find(attrs={"data-testid": "stMetric"}) is not None
    )


def _walk(node: Tag, collector: _Collector) -> None:
    for child in node.children:
        if not isinstance(child, Tag):
            continue

        name = child.name

        if _is_metric(child):
            collector.add("metric", _metric_block_text(child))
            continue

        if name in _HEADING_TAGS:
            text = _collapse(child.get_text(" ", strip=True))
            if text and not _is_boilerplate(text):
                collector.section = text
                collector.add("heading", text)
            continue

        if name == "table":
            for row_text in _table_rows(child):
                collector.add("table_row", row_text)
            continue

        if name in ("ul", "ol"):
            collector.add("list", _list_block_text(child))
            continue

        if name == "pre":
            collector.add("code", child.get_text().strip())
            continue

        # Streamlit 요소 경계에서 조각을 하나로 묶는다 (위 _ELEMENT_CONTAINERS 주석).
        if child.get("data-testid") in _ELEMENT_CONTAINERS and not _has_structured_content(
            child
        ):
            text = _collapse(child.get_text(" ", strip=True))
            if _looks_like_heading(text):
                collector.section = text
                collector.add("heading", text)
            else:
                collector.add("paragraph", text)
            continue

        if name in ("p", "blockquote", "figcaption", "dd", "dt"):
            collector.add("paragraph", _collapse(child.get_text(" ", strip=True)))
            continue

        # 자식 요소가 없는 잎 노드는 그 자체가 텍스트 블록이다.
        # (Streamlit 은 값을 <p> 없이 div/span 에 그대로 넣는 경우가 있다)
        if child.find(True) is None:
            collector.add("paragraph", _collapse(child.get_text(" ", strip=True)))
            continue

        _walk(child, collector)


def _content_root(soup: BeautifulSoup) -> Tag | None:
    for selector in _CONTENT_SELECTORS:
        root = soup.select_one(selector)
        if root is not None:
            return root
    return soup.body


def extract_blocks(html: str) -> ExtractResult:
    """렌더링된 HTML -> 의미 블록. 네트워크를 쓰지 않는 순수 함수다."""
    soup = BeautifulSoup(html, "html.parser")

    warnings: list[str] = []

    # st.dataframe 은 canvas 기반 그리드라 DOM 에 텍스트가 없다. 조용히 비어버리는
    # 대신 경고로 드러낸다 — 표 데이터가 여기 들어있으면 st.table 로 바꿔야 한다.
    if soup.select_one('[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"]'):
        warnings.append(
            "st.dataframe(캔버스 렌더링) 위젯이 있습니다. 이 표의 내용은 DOM 에서 "
            "추출할 수 없으므로 원본 앱에서 st.table/st.markdown 으로 노출해야 합니다."
        )

    for selector in _DROP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    root = _content_root(soup)
    if root is None:
        return ExtractResult(blocks=(), warnings=tuple(warnings), coverage=0.0)

    collector = _Collector()
    _walk(root, collector)

    # 표 행에 헤더를 복제해 붙이므로 captured 가 원문보다 커질 수 있다. 이 값은
    # "빠뜨린 게 있나"를 보는 신호일 뿐이므로 1.0 에서 자른다.
    total_chars = len(_squeeze(root.get_text(" ", strip=True)))
    captured = sum(len(_squeeze(b.text)) for b in collector.blocks)
    coverage = min(1.0, captured / total_chars) if total_chars else 0.0

    # 블록으로 못 잡은 텍스트가 절반을 넘으면 DOM 구조가 바뀐 신호다 (PRD R5).
    if total_chars and coverage < 0.5:
        warnings.append(
            f"본문의 {coverage:.0%} 만 블록으로 추출되었습니다. "
            f"대상 페이지 DOM 구조가 바뀌었을 수 있습니다."
        )

    logger.info(
        "블록 %d개 추출 (유형: %s), 커버리지 %.0f%%",
        len(collector.blocks),
        ", ".join(
            f"{t}={sum(1 for b in collector.blocks if b.block_type == t)}"
            for t in BLOCK_TYPES
            if any(b.block_type == t for b in collector.blocks)
        )
        or "없음",
        coverage * 100,
    )
    for message in warnings:
        logger.warning("%s", message)

    return ExtractResult(
        blocks=tuple(collector.blocks),
        warnings=tuple(warnings),
        coverage=coverage,
    )


# --------------------------------------------------------------------------- #
# 수집 (PRD F1)
# --------------------------------------------------------------------------- #


def robots_allows(url: str, user_agent: str, *, timeout: float = 5.0) -> bool:
    """robots.txt 확인 (PRD F1.8). 읽을 수 없으면 허용으로 간주하고 로그만 남긴다."""
    robots_url = urljoin(url, "/robots.txt")
    try:
        request = Request(robots_url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8", "replace")
    except Exception as exc:
        logger.debug("robots.txt 를 읽지 못해 허용으로 간주합니다 (%s): %s", robots_url, exc)
        return True

    parser = RobotFileParser()
    parser.parse(body.splitlines())
    allowed = parser.can_fetch(user_agent, url)
    if not allowed:
        logger.warning("robots.txt 가 수집을 허용하지 않습니다: %s", url)
    return allowed


def _raise_if_auth(current_url: str) -> None:
    parsed = urlparse(current_url)
    if "/-/auth/" in parsed.path or parsed.netloc == "share.streamlit.io":
        raise AuthRequiredError(
            "로그인이 필요한 페이지는 지원하지 않습니다 "
            f"(인증 페이지로 이동됨: {current_url})."
        )


def _raise_if_known_failure(text: str, url: str) -> None:
    lowered = text.lower()
    if any(marker in lowered for marker in _ASLEEP_MARKERS):
        raise AppAsleepError(
            "대상 앱이 슬립 상태입니다. 브라우저로 한 번 열어 깨운 뒤 다시 시도하세요 "
            f"({url})."
        )
    if any(marker in lowered for marker in _APP_ERROR_MARKERS):
        raise WebLoadError(f"대상 앱이 오류 상태입니다 ({url}): {_collapse(text)[:200]}")


def _same_origin(a: str, b: str) -> bool:
    left, right = urlparse(a), urlparse(b)
    return (left.scheme, left.netloc) == (right.scheme, right.netloc)


def _candidate_frames(page, url: str) -> list:
    """본문이 들어있을 수 있는 프레임 목록.

    **Streamlit Community Cloud 는 앱을 iframe(`//<host>/~/+/`) 안에서 돌린다.**
    최상위 문서는 배지와 제작자 아바타뿐인 껍데기이므로, 최상위만 읽으면 본문이
    0자로 나온다. 반대로 크로스 오리진 프레임(상태 페이지 위젯 등)은 대상 페이지의
    내용이 아니므로 반드시 배제한다.
    """
    frames = [page.main_frame]
    for frame in page.frames:
        if frame is page.main_frame:
            continue
        if frame.url and _same_origin(frame.url, url):
            frames.append(frame)
    return frames


def _frame_text(frame) -> str:
    try:
        return frame.evaluate("() => document.body ? document.body.innerText : ''") or ""
    except Exception:  # 네비게이션 중이면 일시적으로 실패할 수 있다
        return ""


def _wait_for_content(page, url: str, deadline: float, min_chars: int):
    """본문이 가장 많은 프레임을 골라, 길이가 안정될 때까지 폴링한다 (PRD F1.3).

    Streamlit 은 websocket 을 계속 열어두므로 `networkidle` 이 끝나지 않는다.
    따라서 렌더링 완료 판정의 주된 근거는 이 텍스트 길이 폴링이다. 길이가 연속
    2회 같아야 완료로 보는 이유는, 컴포넌트 iframe 이 먼저 채워졌다가 비워지는
    과도 상태에서 잘못된 프레임을 고르지 않기 위함이다.
    """
    best_frame, best_text = page.main_frame, ""
    previous = -1
    stable = 0

    while True:
        frame, text = page.main_frame, ""
        for candidate in _candidate_frames(page, url):
            candidate_text = _frame_text(candidate)
            if len(_squeeze(candidate_text)) > len(_squeeze(text)):
                frame, text = candidate, candidate_text

        if len(_squeeze(text)) > len(_squeeze(best_text)):
            best_frame, best_text = frame, text

        length = len(_squeeze(text))
        if length >= min_chars:
            if length == previous:
                stable += 1
                if stable >= 2:
                    return frame, text
            else:
                stable = 0
            previous = length

        if time.monotonic() >= deadline:
            return best_frame, best_text

        page.wait_for_timeout(500)


def render_html(
    url: str,
    *,
    timeout_sec: int = 30,
    min_content_chars: int = 200,
    user_agent: str = "ExchangeRateRAG/1.0",
) -> RenderedPage:
    """헤드리스 브라우저로 렌더링한다 (PRD F1.2)."""
    try:
        from playwright.sync_api import (
            Error as PlaywrightError,
            TimeoutError as PlaywrightTimeout,
            sync_playwright,
        )
    except ImportError as exc:
        raise BrowserUnavailableError(
            "playwright 가 설치되지 않았습니다. "
            "`python -m pip install -r requirements.txt` 실행 후 "
            "`python -m playwright install chromium` 으로 브라우저를 받으세요."
        ) from exc

    deadline = time.monotonic() + timeout_sec
    logger.info("렌더링 시작 (최대 %d초): %s", timeout_sec, url)

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            raise BrowserUnavailableError(
                "chromium 을 실행할 수 없습니다. "
                "`python -m playwright install chromium` 을 실행하세요."
            ) from exc

        try:
            context = browser.new_context(
                user_agent=user_agent,
                locale="ko-KR",
                viewport={"width": 1440, "height": 2200},
            )
            page = context.new_page()

            try:
                page.goto(
                    url, wait_until="domcontentloaded", timeout=timeout_sec * 1000
                )
            except PlaywrightTimeout as exc:
                raise RenderTimeoutError(
                    f"{timeout_sec}초 안에 페이지가 열리지 않았습니다: {url}"
                ) from exc

            _raise_if_auth(page.url)

            # Streamlit 의 websocket 때문에 도달하지 않는 것이 정상이다. 짧게만 기다린다.
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            try:
                page.wait_for_load_state(
                    "networkidle", timeout=min(3000, remaining_ms) or 1
                )
            except PlaywrightTimeout:
                logger.debug("networkidle 미도달 (Streamlit 에서는 정상)")

            frame, text = _wait_for_content(page, url, deadline, min_content_chars)
            html = frame.content()
            frame_url = frame.url
            final_url = page.url
        finally:
            browser.close()

    _raise_if_auth(final_url)
    _raise_if_known_failure(text, url)

    if len(_squeeze(text)) < min_content_chars:
        raise EmptyContentError(
            f"페이지에서 텍스트를 추출하지 못했습니다 "
            f"({len(_squeeze(text))}자 < {min_content_chars}자). "
            f"JS 렌더링이 끝나지 않았거나 앱이 응답하지 않는 상태로 보입니다: {url}"
        )

    logger.info(
        "렌더링 완료: %d자 (본문 프레임: %s)", len(_squeeze(text)), frame_url
    )
    return RenderedPage(
        html=html, final_url=final_url, frame_url=frame_url, text=text
    )


def load_html(
    html: str,
    *,
    url: str,
    final_url: str | None = None,
    frame_url: str = "",
    text: str = "",
    crawled_at: str | None = None,
) -> PageSnapshot:
    """저장된 스냅샷 HTML 을 그대로 파싱한다 (브라우저 없이 파서만 검증할 때)."""
    result = extract_blocks(html)
    if not result.blocks:
        raise EmptyContentError("HTML 에서 의미 블록을 하나도 추출하지 못했습니다.")
    return PageSnapshot(
        url=url,
        final_url=final_url or url,
        crawled_at=crawled_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        html=html,
        text=text,
        blocks=result.blocks,
        warnings=result.warnings,
        frame_url=frame_url,
    )


def load_url(
    url: str,
    *,
    timeout_sec: int = 30,
    min_content_chars: int = 200,
    user_agent: str = "ExchangeRateRAG/1.0",
    respect_robots: bool = True,
) -> PageSnapshot:
    """URL 을 렌더링해 블록까지 만든 스냅샷을 돌려준다.

    실패는 모두 `WebLoadError` 하위 예외로 나온다 — 호출자(M4 freshness)가
    직전 성공 스냅샷으로 폴백할 수 있게 하기 위함이다 (PRD F1.6).
    """
    if respect_robots and not robots_allows(url, user_agent):
        raise WebLoadError(f"robots.txt 가 이 URL 의 수집을 허용하지 않습니다: {url}")

    rendered = render_html(
        url,
        timeout_sec=timeout_sec,
        min_content_chars=min_content_chars,
        user_agent=user_agent,
    )
    return load_html(
        rendered.html,
        url=url,
        final_url=rendered.final_url,
        frame_url=rendered.frame_url,
        text=rendered.text,
    )
