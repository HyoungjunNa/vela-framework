"""네이버 증권 API 클라이언트 — 실시간 주가/밸류에이션/수급 데이터

인증 불필요. requests만 사용하므로 어디서든 동작 (HF Spaces 포함).
m.stock.naver.com/api 엔드포인트 활용.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://m.stock.naver.com/api/stock"
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
