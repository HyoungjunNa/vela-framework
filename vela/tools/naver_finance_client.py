"""네이버 증권 API 클라이언트 — 실시간 주가/밸류에이션/수급/OHLCV 데이터

인증 불필요. requests + stdlib만 사용하므로 HF Spaces 포함 어디서든 동작.

엔드포인트:
  - m.stock.naver.com/api  : 시세, PER/PBR, 수급, 컨센서스 (JSON)
  - fchart.stock.naver.com : OHLCV 캔들 데이터 (XML) ← pykrx naver 포크
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://m.stock.naver.com/api/stock"
_FCHART_URL = "http://fchart.stock.naver.com/sise.nhn"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 5


def fetch_stock_integration(code: str) -> Optional[Dict[str, Any]]:
    """네이버 증권 통합 데이터 조회 (1회 호출로 시세+밸류에이션+수급+컨센서스)

    Args:
        code: 6자리 종목코드 (예: "005930")

    Returns:
        API 응답 dict 또는 None
    """
    url = f"{_BASE_URL}/{code}/integration"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"[NaverFinance] {code} 조회 실패: {e}")
        return None


def parse_total_infos(data: Dict[str, Any]) -> Dict[str, str]:
    """totalInfos에서 주요 지표 추출

    Returns:
        {"현재가": "181,200", "PER": "37.62배", "PBR": "2.99배", ...}
    """
    result = {}
    for item in data.get("totalInfos", []):
        key = item.get("key", "")
        value = item.get("value", "")
        if key and value:
            result[key] = value
    return result


def parse_deal_trend(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """dealTrendInfos에서 최근 1일 투자자별 순매수 추출

    Returns:
        {"foreign": "-4,715,928", "institution": "+556,164", "individual": "+3,099,928", ...}
    """
    trends: List[Dict] = data.get("dealTrendInfos", [])
    if not trends:
        return None

    latest = trends[0]
    return {
        "foreign": latest.get("foreignerPureBuyQuant", ""),
        "foreign_hold_ratio": latest.get("foreignerHoldRatio", ""),
        "institution": latest.get("organPureBuyQuant", ""),
        "individual": latest.get("individualPureBuyQuant", ""),
        "date": latest.get("bizdate", ""),
        "close_price": latest.get("closePrice", ""),
        "volume": latest.get("accumulatedTradingVolume", ""),
    }


def parse_consensus(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """consensusInfo에서 목표주가 컨센서스 추출

    Returns:
        {"target_price": "216,417", "rating": "4.00"}
    """
    ci = data.get("consensusInfo")
    if not ci:
        return None

    return {
        "target_price": ci.get("priceTargetMean", ""),
        "rating": ci.get("recommMean", ""),
    }


# =============================================================================
# OHLCV 캔들 데이터 (pykrx naver 포크)
# fchart.stock.naver.com/sise.nhn — XML, 인증 불필요
# =============================================================================


def fetch_ohlcv(
    ticker: str,
    count: int = 30,
    timeframe: str = "day",
) -> List[Dict[str, Any]]:
    """네이버 금융 fchart에서 OHLCV 캔들 데이터 조회

    pykrx website/naver/core.py + wrap.py 포크.
    pandas 의존성 없이 dict list 반환.

    Args:
        ticker   : 종목코드 (예: "005930")
        count    : 조회할 캔들 수 (기본 30)
        timeframe: "day" | "week" | "month"

    Returns:
        [{"date": "2026-02-17", "open": 80000, "high": 81000,
          "low": 79500, "close": 80500, "volume": 12345678,
          "change_pct": 0.62}, ...]
        실패 시 빈 리스트
    """
    try:
        resp = requests.get(
            _FCHART_URL,
            params={
                "symbol": ticker,
                "timeframe": timeframe,
                "count": count,
                "requestType": "0",
            },
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_sise_xml(resp.text)
    except Exception as e:
        logger.warning(f"[NaverFchart] {ticker} OHLCV 조회 실패: {e}")
        return []


def _parse_sise_xml(xml_text: str) -> List[Dict[str, Any]]:
    """fchart XML → OHLCV dict list

    XML 형식:
        <item data="20260217|80000|81000|79500|80500|12345678" />
        필드 순서: 날짜|시가|고가|저가|종가|거래량
    """
    result: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
        for node in root.iter("item"):
            row = node.get("data", "")
            parts = row.split("|")
            if len(parts) < 6:
                continue
            date_str, open_, high, low, close, volume = parts[:6]
            try:
                result.append(
                    {
                        "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                        "open": int(open_),
                        "high": int(high),
                        "low": int(low),
                        "close": int(close),
                        "volume": int(volume),
                        "change_pct": 0.0,
                    }
                )
            except (ValueError, IndexError):
                continue
    except ET.ParseError as e:
        logger.warning(f"[NaverFchart] XML 파싱 실패: {e}")
        return []

    # 등락률 계산 (전일 대비)
    for i in range(1, len(result)):
        prev = result[i - 1]["close"]
        curr = result[i]["close"]
        if prev > 0:
            result[i]["change_pct"] = round((curr - prev) / prev * 100, 2)

    return result


def fetch_price_summary(ticker: str, days: int = 5) -> Dict[str, Any]:
    """최근 N일 주가 요약 — EOD 리포트 컨텍스트용

    Args:
        ticker: 종목코드
        days  : 요약 기간 (기본 5일)

    Returns:
        {
          "ticker": "005930",
          "as_of": "2026-02-17",
          "close": 80500,
          "change_pct_1d": 0.62,
          "change_pct_5d": -1.23,
          "avg_volume_5d": 12000000,
          "high_5d": 82000,
          "low_5d": 79000,
        }
        실패 시 빈 dict
    """
    ohlcv = fetch_ohlcv(ticker, count=max(days + 10, 40))
    if not ohlcv:
        return {}

    recent = ohlcv[-days:] if len(ohlcv) >= days else ohlcv
    if not recent:
        return {}

    latest = recent[-1]
    oldest = recent[0]
    avg_volume = int(sum(r["volume"] for r in recent) / len(recent))
    period_change = (
        round((latest["close"] - oldest["close"]) / oldest["close"] * 100, 2)
        if oldest["close"] > 0
        else 0.0
    )

    return {
        "ticker": ticker,
        "as_of": latest["date"],
        "close": latest["close"],
        "change_pct_1d": latest.get("change_pct", 0.0),
        f"change_pct_{days}d": period_change,
        "avg_volume_5d": avg_volume,
        "high_5d": max(r["high"] for r in recent),
        "low_5d": min(r["low"] for r in recent),
    }
