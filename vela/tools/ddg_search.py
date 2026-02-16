"""DuckDuckGo Search Tool"""

from typing import Dict, List
from dataclasses import dataclass
from ddgs import DDGS


@dataclass
class DDGSearchResult:
    title: str
    url: str
    body: str
    date: str
    source: str


class DDGSearchTool:
    """DuckDuckGo를 사용한 뉴스/웹 검색 도구"""

    def __init__(self):
        self.ddgs = DDGS()

    def search_news(self, query: str, max_results: int = 10) -> List[DDGSearchResult]:
        """
        뉴스 검색

        Args:
            query: 검색어
            max_results: 최대 결과 수

        Returns:
            검색 결과 리스트
        """
        try:
            results = list(self.ddgs.news(query, max_results=max_results))
            return [
                DDGSearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    body=r.get("body", ""),
                    date=r.get("date", ""),
                    source=r.get("source", ""),
                )
                for r in results
            ]
        except Exception as e:
            print(f"DDG news search error: {e}")
            return []

    def search_web(self, query: str, max_results: int = 10) -> List[DDGSearchResult]:
        """
        웹 검색

        Args:
            query: 검색어
            max_results: 최대 결과 수

        Returns:
            검색 결과 리스트
        """
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            return [
                DDGSearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    body=r.get("body", ""),
                    date="",
                    source="",
                )
                for r in results
            ]
        except Exception as e:
            print(f"DDG web search error: {e}")
            return []

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
                    "body": r.body[:200] if r.body else "",
                    "date": r.date,
                    "source": r.source,
                }
                for r in results
            ],
        }
