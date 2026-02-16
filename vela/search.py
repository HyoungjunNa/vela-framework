"""멀티소스 통합 검색 모듈

- Naver API (실시간 뉴스)
- DuckDuckGo (글로벌 뉴스)
- Naver Finance (주가/밸류에이션/수급) — 인증 불필요
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .config import STOCK_CODE_MAP
from .schemas import Source, SourceType

logger = logging.getLogger(__name__)

# 역방향 매핑 (종목코드 -> 종목명) 자동 생성
_REVERSE_MAP: Dict[str, str] = {}
for _name, _code in STOCK_CODE_MAP.items():
    if _code not in _REVERSE_MAP:
        _REVERSE_MAP[_code] = _name


class ResearchSearchModule:
    """멀티소스 통합 검색 (Naver + DuckDuckGo)"""

    def __init__(
        self,
        enable_naver: bool = True,
        enable_ddg: bool = True,
        enable_stock_data: bool = True,
        max_workers: int = 4,
    ):
        """
        Args:
            enable_naver: Naver API 사용 여부
            enable_ddg: DuckDuckGo 사용 여부
            enable_stock_data: 네이버 증권 주가/수급 데이터 사용 여부
            max_workers: 병렬 처리 워커 수
        """
        self.enable_naver = enable_naver
        self.enable_ddg = enable_ddg
        self.enable_stock_data = enable_stock_data
        self.max_workers = max_workers

        # 검색 도구 초기화
        self._init_tools()

    def _init_tools(self):
        """검색 도구 초기화"""
        # Naver 검색
        self.naver = None
        if self.enable_naver:
            try:
                from .tools.naver_search import NaverSearchTool

                self.naver = NaverSearchTool()
                logger.info("NaverSearchTool 초기화 완료")
            except Exception as e:
                logger.warning(f"NaverSearchTool 초기화 실패: {e}")

        # DuckDuckGo 검색
        self.ddg = None
        if self.enable_ddg:
            try:
                from .tools.ddg_search import DDGSearchTool

                self.ddg = DDGSearchTool()
                logger.info("DDGSearchTool 초기화 완료")
            except Exception as e:
                logger.warning(f"DDGSearchTool 초기화 실패: {e}")

        # 네이버 증권 (주가/밸류에이션/수급) — 인증 불필요
        self.stock_data_available = self.enable_stock_data
        if self.stock_data_available:
            logger.info("네이버 증권 데이터 활성화")

    def search_all(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[str]] = None,
        stock_code: Optional[str] = None,
    ) -> List[Source]:
        """모든 소스에서 병렬 검색

        Args:
            query: 검색 쿼리
            max_results: 소스당 최대 결과 수
            sources: 사용할 소스 목록 (기본: ["naver", "ddg"])
            stock_code: 종목코드 (현재는 사용하지 않지만 인터페이스 호환)

        Returns:
            Source 리스트
        """
        if sources is None:
            sources = ["naver", "ddg"]
            if self.stock_data_available:
                sources.append("stock_data")

        # 종목코드 추출 (없으면 쿼리에서 추출)
        if not stock_code:
            stock_code = self._extract_stock_code(query)

        results = []
        futures = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Naver API
            if "naver" in sources and self.naver:
                futures[executor.submit(self._search_naver, query, max_results)] = (
                    "naver"
                )

            # DuckDuckGo
            if "ddg" in sources and self.ddg:
                futures[executor.submit(self._search_ddg, query, max_results)] = "ddg"

            # 네이버 증권 시세/수급 (종목코드 필수)
            if "stock_data" in sources and self.stock_data_available and stock_code:
                futures[
                    executor.submit(self._search_stock_data, stock_code)
                ] = "stock_data"

            # 결과 수집
            for future in as_completed(futures, timeout=30):
                source_name = futures[future]
                try:
                    source_results = future.result()
                    results.extend(source_results)
                    logger.info(f"{source_name}: {len(source_results)}개 결과")
                except Exception as e:
                    logger.warning(f"{source_name} 검색 실패: {e}")

        # 관련성 점수로 정렬
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results

    def _search_naver(self, query: str, max_results: int) -> List[Source]:
        """Naver API 검색"""
        sources = []
        try:
            results = self.naver.search_news(query, max_results)
            for r in results:
                sources.append(
                    Source(
                        title=r.title,
                        url=r.url,
                        source_type=SourceType.NEWS,
                        date=r.pub_date[:10] if r.pub_date else "",
                        snippet=r.description[:300] if r.description else "",
                        relevance_score=0.8,  # 실시간 뉴스 기본 점수
                    )
                )
        except Exception as e:
            logger.warning(f"Naver 검색 실패: {e}")
        return sources

    def _search_ddg(self, query: str, max_results: int) -> List[Source]:
        """DuckDuckGo 검색"""
        sources = []
        try:
            results = self.ddg.search_news(query, max_results)
            for r in results:
                sources.append(
                    Source(
                        title=r.title,
                        url=r.url,
                        source_type=SourceType.NEWS,
                        date=r.date,
                        snippet=r.body[:300] if r.body else "",
                        relevance_score=0.7,  # 글로벌 뉴스 기본 점수
                    )
                )
        except Exception as e:
            logger.warning(f"DDG 검색 실패: {e}")
        return sources

    def _search_stock_data(self, stock_code: str) -> List[Source]:
        """네이버 증권 통합 데이터 조회 (1회 API 호출 → 시세 + 수급 Source 생성)"""
        from .tools.naver_finance_client import (
            fetch_stock_integration,
            parse_deal_trend,
            parse_total_infos,
        )

        data = fetch_stock_integration(stock_code)
        if not data:
            return []

        stock_name = data.get("stockName") or _REVERSE_MAP.get(stock_code, stock_code)
        results: List[Source] = []

        # 1. 재무지표 Source (extract_confirmed_data 호환 형식)
        infos = parse_total_infos(data)
        if infos:
            parts = []
            if v := infos.get("전일"):
                parts.append(f"현재가: {v}원")
            if v := infos.get("PER"):
                parts.append(f"PER(TTM): {v}")
            if v := infos.get("추정PER"):
                parts.append(f"추정PER: {v}")
            if v := infos.get("PBR"):
                parts.append(f"PBR: {v}")
            if v := infos.get("EPS"):
                parts.append(f"EPS: {v}")
            if v := infos.get("배당수익률"):
                parts.append(f"배당수익률: {v}")
            if v := infos.get("시총"):
                parts.append(f"시총: {v}")

            if parts:
                results.append(
                    Source(
                        title=f"{stock_name} 재무지표 (네이버증권)",
                        url=f"naver://finance/{stock_code}",
                        source_type=SourceType.PRICE,
                        snippet=" | ".join(parts),
                        relevance_score=0.95,
                    )
                )

        # 2. 투자자동향 Source
        trend = parse_deal_trend(data)
        if trend:
            parts = []
            if v := trend.get("foreign"):
                parts.append(f"외국인순매수: {v}주")
            if v := trend.get("institution"):
                parts.append(f"기관순매수: {v}주")
            if v := trend.get("individual"):
                parts.append(f"개인순매수: {v}주")
            if v := trend.get("foreign_hold_ratio"):
                parts.append(f"외인소진율: {v}")

            if parts:
                results.append(
                    Source(
                        title=f"{stock_name} 투자자동향 (네이버증권)",
                        url=f"naver://investor/{stock_code}",
                        source_type=SourceType.INVESTOR,
                        snippet=" | ".join(parts),
                        relevance_score=0.95,
                    )
                )

        return results

    def _extract_stock_code(self, query: str) -> Optional[str]:
        """쿼리에서 종목코드 추출 (config.STOCK_CODE_MAP 기반)

        Args:
            query: 검색 쿼리

        Returns:
            종목코드 또는 None
        """
        # 1. 6자리 숫자 패턴 (직접 코드 입력)
        match = re.search(r"\b(\d{6})\b", query)
        if match:
            return match.group(1)

        # 2. STOCK_CODE_MAP에서 정확히 일치하는 종목명 검색
        # 긴 이름부터 먼저 검색 (삼성바이오로직스 > 삼성)
        sorted_names = sorted(STOCK_CODE_MAP.keys(), key=len, reverse=True)
        for name in sorted_names:
            if name in query:
                return STOCK_CODE_MAP[name]

        return None

    def resolve_stock_code(self, query: str) -> Optional[str]:
        """종목명/쿼리에서 종목코드 추출 (외부 호출용)

        Args:
            query: 종목명 또는 검색 쿼리

        Returns:
            종목코드 또는 None
        """
        return self._extract_stock_code(query)

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """종목코드로 종목명 조회

        Args:
            stock_code: 6자리 종목코드

        Returns:
            종목명 또는 None
        """
        return _REVERSE_MAP.get(stock_code)

    @staticmethod
    def extract_confirmed_data(sources: List[Source]) -> Dict[str, Any]:
        """pykis/fnguide 소스에서 확보된 데이터를 구조화하여 추출

        LLM이 판단하지 않고 시스템이 자동으로 추출.
        이 데이터는 synthesis 프롬프트에 강제 주입됨.

        Args:
            sources: 수집된 소스 리스트

        Returns:
            {
                "valuation": {"per": "7.0배", "pbr": "1.2배", ...},
                "price": {"current": "68,000원", "change": "+2.5%"},
                "investor": {"foreign": "+1,234주", "institution": "-567주"},
                "business": "반도체 및 관련장비 제조업...",
                "provided_fields": ["per", "pbr", "eps", "roe", ...],
            }
        """
        confirmed = {
            "valuation": {},
            "price": {},
            "investor": {},
            "business": "",
            "provided_fields": [],
        }

        for src in sources:
            snippet = src.snippet or ""
            title = src.title or ""

            # 1. 재무지표 파싱 (pykis://financial)
            if "재무지표" in title or "밸류에이션" in title:
                # 현재가
                if match := re.search(r"현재가[:\s]+([0-9,]+)원", snippet):
                    confirmed["valuation"]["current_price"] = match.group(1) + "원"
                    confirmed["provided_fields"].append("current_price")

                # 12M FWD PER
                if match := re.search(r"12M FWD PER[:\s]+([0-9.]+)배", snippet):
                    confirmed["valuation"]["12m_fwd_per"] = match.group(1) + "배"
                    confirmed["provided_fields"].append("12m_fwd_per")
                elif match := re.search(r"PER\(TTM\)[:\s]+([0-9.]+)배", snippet):
                    confirmed["valuation"]["per_ttm"] = match.group(1) + "배"
                    confirmed["provided_fields"].append("per_ttm")

                # PBR
                if match := re.search(r"PBR[:\s]+([0-9.]+)배", snippet):
                    confirmed["valuation"]["pbr"] = match.group(1) + "배"
                    confirmed["provided_fields"].append("pbr")

                # 12M FWD EPS
                if match := re.search(r"12M FWD EPS[:\s]+([0-9,]+)원", snippet):
                    confirmed["valuation"]["12m_fwd_eps"] = match.group(1) + "원"
                    confirmed["provided_fields"].append("12m_fwd_eps")

                # ROE
                if match := re.search(r"ROE[:\s]+([0-9.]+)%", snippet):
                    confirmed["valuation"]["roe"] = match.group(1) + "%"
                    confirmed["provided_fields"].append("roe")

            # 2. 실시간 시세 파싱 (pykis://quote)
            elif "실시간시세" in title:
                if match := re.search(r"현재가\s+([0-9,]+)원", snippet):
                    confirmed["price"]["current"] = match.group(1) + "원"
                if match := re.search(r"전일대비\s+([+-]?[0-9.]+)%", snippet):
                    confirmed["price"]["change"] = match.group(1) + "%"

            # 3. 투자자동향 파싱 (pykis://investor)
            elif "투자자동향" in title or "수급" in title:
                if match := re.search(r"외국인순매수[:\s]+([+-]?[0-9,]+)주", snippet):
                    confirmed["investor"]["foreign_net"] = match.group(1) + "주"
                    confirmed["provided_fields"].append("foreign_net")
                if match := re.search(r"기관순매수[:\s]+([+-]?[0-9,]+)주", snippet):
                    confirmed["investor"]["institution_net"] = match.group(1) + "주"
                    confirmed["provided_fields"].append("institution_net")

            # 4. 사업개요 파싱 (fnguide)
            elif "사업개요" in title:
                confirmed["business"] = snippet[:200] if snippet else ""
                confirmed["provided_fields"].append("business_summary")

        # 중복 제거
        confirmed["provided_fields"] = list(set(confirmed["provided_fields"]))

        return confirmed
