"""Naver Search API Tool - 페이지네이션 및 날짜 필터링 지원"""

import os
import re
import random
import requests
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

# .env 로드
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class NaverSearchResult:
    title: str
    url: str
    description: str
    pub_date: str


class NaverSearchTool:
    """Naver 검색 API를 사용한 뉴스 검색 도구

    Features:
        - 페이지네이션 지원 (최대 1000개)
        - 날짜 범위 필터링 (startDate, endDate)
        - API 키 순환 (Rate limit 대응)
        - URL 기반 중복 제거
    """

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms

    def _load_api_keys(self) -> List[Tuple[str, str]]:
        """환경변수에서 API 키 로드"""
        keys = []
        for i in range(1, 10):
            client_id = os.getenv(f"NAVER_CLIENT_ID_{i}")
            client_secret = os.getenv(f"NAVER_CLIENT_SECRET_{i}")
            if client_id and client_secret:
                keys.append((client_id, client_secret))
        return keys

    def _get_headers(self, key_index: Optional[int] = None) -> Dict[str, str]:
        """API 키로 헤더 생성

        Args:
            key_index: 특정 키 인덱스 (None이면 랜덤)
        """
        if not self.api_keys:
            raise ValueError("No Naver API keys configured")

        if key_index is not None and 0 <= key_index < len(self.api_keys):
            client_id, client_secret = self.api_keys[key_index]
        else:
            client_id, client_secret = random.choice(self.api_keys)

        return {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

    def _request_with_retry(self, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        """401 에러 시 다른 키로 재시도"""
        tried_keys: Set[int] = set()

        for _ in range(max_retries):
            # 아직 시도하지 않은 키 선택
            available = [i for i in range(len(self.api_keys)) if i not in tried_keys]
            if not available:
                break

            key_idx = random.choice(available)
            tried_keys.add(key_idx)

            try:
                self._rate_limit()
                response = requests.get(
                    self.base_url,
                    headers=self._get_headers(key_idx),
                    params=params,
                    timeout=10,
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    print(f"키 {key_idx + 1} 인증 실패, 다른 키로 재시도...")
                    continue
                elif response.status_code == 429:
                    print("Rate limit, 1초 대기...")
                    time.sleep(1)
                    continue
                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException as e:
                print(f"요청 실패: {e}")

        return None

    def _rate_limit(self):
        """요청 간격 관리"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _normalize_date(self, date_str: str) -> str:
        """날짜를 YYYYMMDD 형식으로 정규화"""
        if not date_str:
            return ""

        # 이미 YYYYMMDD 형식인 경우
        if re.match(r"^\d{8}$", date_str):
            return date_str

        # YYYY-MM-DD 형식인 경우
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str.replace("-", "")

        return date_str

    def _parse_rfc2822_date(self, date_str: str) -> Optional[datetime]:
        """RFC 2822 날짜 파싱 (예: 'Wed, 01 Jan 2025 10:30:00 +0900')"""
        if not date_str:
            return None
        try:
            return parsedate_to_datetime(date_str)
        except Exception:
            return None

    def _clean_html(self, text: str) -> str:
        """HTML 태그 및 엔티티 제거"""
        if not text:
            return ""
        # 태그 제거
        text = re.sub(r"<[^>]+>", "", text)
        # HTML 엔티티 처리
        text = text.replace("&quot;", '"').replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("<b>", "").replace("</b>", "")
        return text.strip()

    def search_news(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "date",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[NaverSearchResult]:
        """
        뉴스 검색 (단일 페이지)

        Args:
            query: 검색어
            max_results: 최대 결과 수 (1-100)
            sort: 정렬 방식 (sim: 유사도, date: 날짜)
            start_date: 시작 날짜 (YYYY-MM-DD 또는 YYYYMMDD)
            end_date: 종료 날짜 (YYYY-MM-DD 또는 YYYYMMDD)

        Returns:
            검색 결과 리스트
        """
        if not self.api_keys:
            return []

        self._rate_limit()

        params = {"query": query, "display": min(max_results, 100), "sort": sort}

        # Note: 네이버 뉴스 API는 startDate/endDate 파라미터를 지원하지 않음
        # 날짜 필터링은 결과를 받은 후 pubDate 기반으로 수행

        data = self._request_with_retry(params)
        if not data:
            return []

        results = []
        for item in data.get("items", []):
            results.append(
                NaverSearchResult(
                    title=self._clean_html(item.get("title", "")),
                    url=item.get("link", ""),
                    description=self._clean_html(item.get("description", "")),
                    pub_date=item.get("pubDate", ""),
                )
            )
        return results

    def _is_in_date_range(
        self, pub_date: str, start_date: Optional[str], end_date: Optional[str]
    ) -> bool:
        """pubDate가 날짜 범위 내에 있는지 확인"""
        if not start_date and not end_date:
            return True

        parsed = self._parse_rfc2822_date(pub_date)
        if not parsed:
            return True  # 파싱 실패시 포함

        pub_date_str = parsed.strftime("%Y%m%d")

        if start_date and pub_date_str < start_date:
            return False
        return not (end_date and pub_date_str > end_date)

    def search_news_paginated(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_pages: int = 10,
        items_per_page: int = 100,
        sort: str = "date",
    ) -> List[NaverSearchResult]:
        """
        페이지네이션을 지원하는 뉴스 검색

        Args:
            query: 검색어
            start_date: 시작 날짜 (YYYY-MM-DD 또는 YYYYMMDD)
            end_date: 종료 날짜 (YYYY-MM-DD 또는 YYYYMMDD)
            max_pages: 최대 페이지 수
            items_per_page: 페이지당 아이템 수 (최대 100)
            sort: 정렬 옵션 (date: 날짜순, sim: 유사도순)

        Returns:
            모든 페이지의 검색 결과 리스트

        Note:
            네이버 뉴스 API는 startDate/endDate 파라미터를 지원하지 않음.
            pubDate 기반 필터링으로 날짜 범위를 적용함.
        """
        if not self.api_keys:
            return []

        all_results: List[NaverSearchResult] = []
        seen_urls: Set[str] = set()

        # 날짜 정규화
        normalized_start = self._normalize_date(start_date) if start_date else None
        normalized_end = self._normalize_date(end_date) if end_date else None

        # 날짜 범위 밖 뉴스 연속 카운터 (조기 종료용)
        out_of_range_count = 0
        max_out_of_range = 50  # 연속 50개가 범위 밖이면 종료

        for page in range(1, max_pages + 1):
            start_index = 1 + (page - 1) * items_per_page

            # 네이버 API는 최대 1000개까지만 지원
            if start_index > 1000:
                print("Naver API 최대 한도 도달 (1000개)")
                break

            self._rate_limit()

            params = {
                "query": query,
                "display": min(items_per_page, 100),
                "start": start_index,
                "sort": sort,
            }

            data = self._request_with_retry(params)
            if not data:
                print(f"페이지 {page} 요청 실패")
                break

            items = data.get("items", [])
            if not items:
                print(f"페이지 {page}에서 결과 없음")
                break

            # 중복 URL 제거 + 날짜 필터링
            new_count = 0
            for item in items:
                url = item.get("link", "")
                pub_date = item.get("pubDate", "")

                # 날짜 범위 체크
                if not self._is_in_date_range(
                    pub_date, normalized_start, normalized_end
                ):
                    out_of_range_count += 1
                    if out_of_range_count >= max_out_of_range:
                        print(
                            f"날짜 범위 밖 뉴스 {max_out_of_range}개 연속 - 검색 종료"
                        )
                        break
                    continue

                out_of_range_count = 0  # 리셋

                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(
                        NaverSearchResult(
                            title=self._clean_html(item.get("title", "")),
                            url=url,
                            description=self._clean_html(item.get("description", "")),
                            pub_date=pub_date,
                        )
                    )
                    new_count += 1

            if out_of_range_count >= max_out_of_range:
                break

            if new_count == 0:
                print(f"페이지 {page}에 새로운 결과 없음 (날짜 필터 적용)")
            else:
                print(f"페이지 {page}: {new_count}개 추가 (총 {len(all_results)}개)")

        return all_results

    def search_news_for_trading_date(
        self,
        query: str,
        trading_date: str,
        trading_days_back: int = 10,
        max_results: int = 50,
    ) -> List[NaverSearchResult]:
        """
        특정 거래일 기준 과거 뉴스 검색 (10영업일)

        Args:
            query: 검색어 (회사명)
            trading_date: 기준 거래일 (YYYY-MM-DD)
            trading_days_back: 검색할 영업일 수 (기본 10일)
            max_results: 최대 결과 수

        Returns:
            날짜 필터링된 검색 결과
        """
        # 10영업일 ≈ 14일 (주말 제외)
        calendar_days = int(trading_days_back * 1.5)

        # 날짜 계산
        try:
            end_dt = datetime.strptime(trading_date, "%Y-%m-%d")
        except ValueError:
            end_dt = datetime.now()

        start_dt = end_dt - timedelta(days=calendar_days)

        start_date = start_dt.strftime("%Y%m%d")
        end_date = end_dt.strftime("%Y%m%d")

        print(f"검색 범위: {start_date} ~ {end_date} ({calendar_days}일)")

        # 필요한 페이지 수 계산
        max_pages = (max_results // 100) + 1

        results = self.search_news_paginated(
            query=query,
            start_date=start_date,
            end_date=end_date,
            max_pages=max_pages,
            items_per_page=100,
            sort="date",
        )

        # 최대 결과 수 제한
        return results[:max_results]

    def search_news_dict(self, query: str, max_results: int = 5) -> Dict:
        """MCP 호환 딕셔너리 형식으로 검색 결과 반환"""
        results = self.search_news(query, max_results)
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "description": r.description[:200] if r.description else "",
                    "date": r.pub_date,
                }
                for r in results
            ],
        }


if __name__ == "__main__":
    # 테스트
    tool = NaverSearchTool()
    print(f"API 키: {len(tool.api_keys)}개")

    # 페이지네이션 테스트
    print("\n=== 페이지네이션 테스트 ===")
    results = tool.search_news_for_trading_date(
        query="삼성전자",
        trading_date="2025-12-30",
        trading_days_back=10,
        max_results=20,
    )

    print(f"\n총 {len(results)}개 결과:")
    for r in results[:5]:
        print(f"  - [{r.pub_date[:16]}] {r.title[:50]}...")
