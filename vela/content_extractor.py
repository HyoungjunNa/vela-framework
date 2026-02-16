"""콘텐츠 추출기 - 웹 페이지 및 PDF 콘텐츠 추출

- 뉴스 기사 본문 추출 (requests + BeautifulSoup)
- PDF 텍스트 추출 (pypdf)
- 증권사 리포트 요약 추출
"""

import io
import logging
import re
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ContentExtractor:
    """웹 페이지 및 PDF 콘텐츠 추출기"""

    # 뉴스 사이트별 본문 선택자
    NEWS_SELECTORS = {
        "n.news.naver.com": {
            "content": "#dic_area, #newsct_article, .newsct_article",
            "title": ".media_end_head_headline, #articleTitle",
        },
        "news.naver.com": {
            "content": "#articeBody, #newsEndContents",
            "title": ".article_head_title",
        },
        "www.hankyung.com": {
            "content": "#articletxt, .article-body",
            "title": ".headline",
        },
        "www.mk.co.kr": {
            "content": "#article_body, .art_txt",
            "title": ".top_title",
        },
        "www.sedaily.com": {
            "content": "#v-left-scroll-in, .article_view",
            "title": ".article_tit",
        },
        "www.edaily.co.kr": {
            "content": "#contents, .news_body",
            "title": ".news_title",
        },
        "www.yna.co.kr": {
            "content": "#articleWrap, .story-news",
            "title": ".tit-article",
        },
        # 기본 선택자
        "default": {
            "content": "article, .article-body, .article-content, .post-content, main",
            "title": "h1, .title, .headline",
        },
    }

    def __init__(
        self,
        timeout: int = 10,
        max_content_length: int = 10000,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    ):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.headers = {"User-Agent": user_agent}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def extract(self, url: str) -> Optional[str]:
        """URL에서 콘텐츠 추출 (자동 타입 감지)

        Args:
            url: 추출할 URL

        Returns:
            추출된 텍스트 또는 None
        """
        if not url:
            return None

        try:
            # PDF 파일인 경우
            if url.lower().endswith(".pdf") or "pdf" in url.lower():
                return self.extract_pdf_content(url)

            # 웹 페이지인 경우
            return self.extract_news_body(url)

        except Exception as e:
            logger.warning(f"콘텐츠 추출 실패: {url} - {e}")
            return None

    def extract_news_body(self, url: str) -> Optional[str]:
        """뉴스 기사 본문 추출

        Args:
            url: 뉴스 URL

        Returns:
            본문 텍스트 또는 None
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # 인코딩 처리
            if response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, "html.parser")

            # 불필요한 요소 제거
            for tag in soup.find_all(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()

            # 도메인별 선택자 사용
            domain = urlparse(url).netloc
            selectors = self.NEWS_SELECTORS.get(domain, self.NEWS_SELECTORS["default"])

            # 본문 추출
            content = None
            for selector in selectors["content"].split(", "):
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(separator="\n", strip=True)
                    break

            if not content:
                # Fallback: 가장 긴 텍스트 블록 찾기
                paragraphs = soup.find_all("p")
                if paragraphs:
                    content = "\n".join(
                        p.get_text(strip=True)
                        for p in paragraphs
                        if len(p.get_text(strip=True)) > 50
                    )

            if content:
                # 정리
                content = self._clean_text(content)
                return content[: self.max_content_length]

            return None

        except requests.RequestException as e:
            logger.warning(f"뉴스 요청 실패: {url} - {e}")
            return None
        except Exception as e:
            logger.warning(f"뉴스 파싱 실패: {url} - {e}")
            return None

    def extract_pdf_content(self, pdf_url: str) -> Optional[str]:
        """PDF 텍스트 추출

        Args:
            pdf_url: PDF URL

        Returns:
            추출된 텍스트 또는 None
        """
        try:
            # pypdf 임포트 (선택적 의존성)
            try:
                from pypdf import PdfReader
            except ImportError:
                logger.warning("pypdf 미설치 - pip install pypdf")
                return None

            # PDF 다운로드
            response = self.session.get(pdf_url, timeout=30)
            response.raise_for_status()

            # PDF 파싱
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)

            # 텍스트 추출 (최대 10페이지)
            text_parts = []
            for i, page in enumerate(reader.pages[:10]):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            if text_parts:
                content = "\n\n".join(text_parts)
                content = self._clean_text(content)
                return content[: self.max_content_length]

            return None

        except requests.RequestException as e:
            logger.warning(f"PDF 다운로드 실패: {pdf_url} - {e}")
            return None
        except Exception as e:
            logger.warning(f"PDF 파싱 실패: {pdf_url} - {e}")
            return None

    def extract_report_summary(self, url: str) -> Dict:
        """증권사 리포트 요약 추출

        Args:
            url: 리포트 URL

        Returns:
            추출된 정보 딕셔너리
        """
        result = {
            "title": "",
            "securities_firm": "",
            "target_price": None,
            "rating": "",
            "summary": "",
            "content": "",
        }

        try:
            content = self.extract(url)
            if not content:
                return result

            result["content"] = content

            # 목표주가 추출 (정규식)
            price_match = re.search(r"목표\s*주가[:\s]*([0-9,]+)\s*원", content)
            if price_match:
                result["target_price"] = int(price_match.group(1).replace(",", ""))

            # 투자의견 추출
            for rating in [
                "매수",
                "Buy",
                "중립",
                "Neutral",
                "매도",
                "Sell",
                "보유",
                "Hold",
            ]:
                if rating.lower() in content.lower():
                    result["rating"] = rating
                    break

            # 증권사 추출 (URL 기반)
            if "miraeasset" in url.lower():
                result["securities_firm"] = "미래에셋증권"
            elif "samsung" in url.lower():
                result["securities_firm"] = "삼성증권"
            elif "kb" in url.lower():
                result["securities_firm"] = "KB증권"
            elif "shinhan" in url.lower():
                result["securities_firm"] = "신한투자증권"
            elif "naver" in url.lower():
                # 네이버 증권에서 증권사명 추출
                firm_match = re.search(r"([가-힣]+증권)", content[:200])
                if firm_match:
                    result["securities_firm"] = firm_match.group(1)

            # 요약 (첫 500자)
            result["summary"] = content[:500]

            return result

        except Exception as e:
            logger.warning(f"리포트 요약 추출 실패: {url} - {e}")
            return result

    def _clean_text(self, text: str) -> str:
        """텍스트 정리

        - 연속 공백 제거
        - 불필요한 문자 제거
        - 정규화
        """
        if not text:
            return ""

        # 연속 공백/줄바꿈 정리
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # HTML 엔티티 정리
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")

        # 특수문자 정리
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        return text.strip()

    def batch_extract(
        self, urls: list, max_concurrent: int = 5
    ) -> Dict[str, Optional[str]]:
        """여러 URL에서 일괄 추출

        Args:
            urls: URL 리스트
            max_concurrent: 최대 동시 요청 수 (현재는 순차 처리)

        Returns:
            {url: content} 딕셔너리
        """
        results = {}
        for url in urls:
            results[url] = self.extract(url)
        return results


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = ContentExtractor()

    # 테스트 URL
    test_urls = [
        "https://n.news.naver.com/mnews/article/003/0013689587?sid=101",  # 네이버 뉴스
    ]

    for url in test_urls:
        print(f"\n=== {url} ===")
        content = extractor.extract(url)
        if content:
            print(f"추출 성공: {len(content)}자")
            print(content[:500] + "...")
        else:
            print("추출 실패")
