"""AdversaryAgent - Perplexity 기반 검증 에이전트

CoT Research 결과를 검증하고 재생성 트리거를 결정합니다.

사용 예시:
    adversary = AdversaryAgent()
    verification = adversary.verify(research_result)

    if verification.verdict == "revise":
        # 수정 필요
        ...
    elif verification.verdict == "need_more_search":
        # 추가 검색 실행
        for query in verification.suggested_counter_queries:
            ...
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import requests

from .schemas import (
    ClaimEvidence,
    IssueSeverity,
    IssueType,
    ResearchResult,
    VerificationIssue,
    VerificationResult,
    VerificationVerdict,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Perplexity API 클라이언트
# =============================================================================


class PerplexityClient:
    """Perplexity API 클라이언트

    sonar 모델을 사용해 실시간 웹 검색 기반 검증 수행
    """

    API_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonar",
        timeout: int = 60,
    ):
        """
        Args:
            api_key: Perplexity API 키 (없으면 환경변수 PERPLEXITY_API_KEY 사용)
            model: 사용할 모델 (sonar, sonar-pro 등)
            timeout: 요청 타임아웃 (초)
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Perplexity API 키가 필요합니다. "
                "PERPLEXITY_API_KEY 환경변수를 설정하세요."
            )

        self.model = model
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        return_citations: bool = True,
    ) -> Dict:
        """Perplexity API 쿼리 실행

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            return_citations: 인용 URL 반환 여부

        Returns:
            {"content": str, "citations": List[str]}
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "return_citations": return_citations,
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # 응답 파싱
            content = data["choices"][0]["message"]["content"]
            citations = data.get("citations", [])

            return {
                "content": content,
                "citations": citations,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API 요청 실패: {e}")
            raise


# =============================================================================
# Adversary Agent
# =============================================================================


class AdversaryAgent:
    """Adversary Agent - 결과 검증 및 재생성 트리거

    Perplexity sonar-pro를 활용해:
    1. 각 claim의 사실 여부 검증
    2. 오래된 정보 감지
    3. 누락된 핵심 사실 발견
    4. 추가 검색 쿼리 제안
    """

    VERIFICATION_SYSTEM_PROMPT = """당신은 금융/투자 리서치 결과를 검증하는 Adversary Agent입니다.

주어진 리서치 결과의 각 주장(claim)을 최신 정보로 검증하고,
문제가 있으면 구체적인 이슈를 보고하세요.

검증 기준:
1. unsupported_claim: 근거가 불충분하거나 출처가 없는 주장
2. contradiction: 다른 신뢰할 수 있는 소스와 모순되는 정보
3. stale_info: 30일 이상 오래된 정보 (최신 업데이트 있음)
4. missing_key_fact: 해당 주제에서 반드시 언급해야 할 핵심 사실 누락

반드시 JSON 형식으로 응답하세요."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonar",
        strict_mode: bool = False,
    ):
        """
        Args:
            api_key: Perplexity API 키
            model: 사용할 모델
            strict_mode: 엄격 모드 (더 많은 이슈 감지)
        """
        self.client = PerplexityClient(api_key=api_key, model=model)
        self.model = model
        self.strict_mode = strict_mode

    def verify(
        self,
        result: ResearchResult,
        max_claims_to_verify: int = 10,
    ) -> VerificationResult:
        """리서치 결과 검증

        Args:
            result: 검증할 ResearchResult
            max_claims_to_verify: 최대 검증할 claim 수

        Returns:
            VerificationResult
        """
        start_time = time.time()

        # claim_evidence_map에서 claims 추출
        claims = self._extract_claims(result)
        if not claims:
            # claims가 없으면 key_findings 사용
            claims = [
                {"id": i, "claim": f, "evidence": []}
                for i, f in enumerate(result.key_findings)
            ]

        # 검증할 claims 제한
        claims_to_verify = claims[:max_claims_to_verify]

        # 검증 프롬프트 생성
        prompt = self._build_verification_prompt(
            query=result.query,
            conclusion=result.conclusion,
            claims=claims_to_verify,
        )

        # Perplexity 검증 실행
        try:
            response = self.client.query(
                prompt=prompt,
                system_prompt=self.VERIFICATION_SYSTEM_PROMPT,
                return_citations=True,
            )

            # 응답 파싱
            verification = self._parse_verification_response(
                response=response,
                elapsed_ms=int((time.time() - start_time) * 1000),
            )

            logger.info(
                f"검증 완료: verdict={verification.verdict}, "
                f"issues={len(verification.issues)}, "
                f"confidence={verification.confidence:.0%}"
            )

            return verification

        except Exception as e:
            logger.error(f"검증 실패: {e}")
            # 실패 시 기본 accept 반환 (fail-open)
            return VerificationResult(
                verdict=VerificationVerdict.ACCEPT,
                issues=[],
                confidence=0.0,
                elapsed_ms=int((time.time() - start_time) * 1000),
            )

    def _extract_claims(self, result: ResearchResult) -> List[Dict]:
        """ResearchResult에서 claims 추출"""
        claims = []

        # claim_evidence_map이 있으면 사용
        if result.metadata.claim_evidence_map:
            for i, ce in enumerate(result.metadata.claim_evidence_map):
                claims.append(
                    {
                        "id": i,
                        "claim": ce.claim,
                        "evidence": ce.evidence,
                    }
                )

        return claims

    def _build_verification_prompt(
        self,
        query: str,
        conclusion: str,
        claims: List[Dict],
    ) -> str:
        """검증 프롬프트 생성"""

        claims_text = ""
        for c in claims:
            evidence_text = ", ".join(
                e.get("support", "")[:50] for e in c.get("evidence", [])
            )
            claims_text += f"\n[Claim {c['id']}]: {c['claim']}"
            if evidence_text:
                claims_text += f"\n  근거: {evidence_text}..."

        prompt = f"""다음 리서치 결과를 검증하세요:

## 원본 쿼리
{query}

## 결론 요약
{conclusion[:500]}...

## 검증할 Claims
{claims_text}

## 응답 형식 (JSON)
{{
  "verdict": "accept|revise|need_more_search",
  "issues": [
    {{
      "type": "unsupported_claim|contradiction|stale_info|missing_key_fact",
      "severity": "low|medium|high",
      "claim_id": 0,
      "why": "문제 설명",
      "citations": ["검증에 사용한 URL"],
      "suggested_edit": "수정 제안 (있다면)"
    }}
  ],
  "suggested_counter_queries": ["추가 검색이 필요하다면 쿼리 제안"],
  "confidence": 0.85,
  "additional_sources": [
    {{"title": "새로 발견한 소스", "url": "...", "snippet": "..."}}
  ]
}}

최신 정보를 기반으로 각 claim을 검증하고,
발견한 모든 이슈와 추가 소스를 포함해 응답하세요."""

        return prompt

    def _parse_verification_response(
        self,
        response: Dict,
        elapsed_ms: int,
    ) -> VerificationResult:
        """Perplexity 응답 파싱"""

        content = response.get("content", "")
        citations = response.get("citations", [])

        # JSON 추출
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            logger.warning("JSON 응답 파싱 실패, 기본값 반환")
            return VerificationResult(
                verdict=VerificationVerdict.ACCEPT,
                confidence=0.5,
                elapsed_ms=elapsed_ms,
                additional_sources=[
                    {"title": "", "url": url, "snippet": ""} for url in citations
                ],
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 에러: {e}")
            return VerificationResult(
                verdict=VerificationVerdict.ACCEPT,
                confidence=0.5,
                elapsed_ms=elapsed_ms,
            )

        # verdict 파싱
        verdict_str = data.get("verdict", "accept").lower()
        try:
            verdict = VerificationVerdict(verdict_str)
        except ValueError:
            verdict = VerificationVerdict.ACCEPT

        # issues 파싱
        issues = []
        for issue_data in data.get("issues", []):
            try:
                issue_type = IssueType(issue_data.get("type", "unsupported_claim"))
                severity = IssueSeverity(issue_data.get("severity", "low"))

                issue = VerificationIssue(
                    type=issue_type,
                    severity=severity,
                    claim_id=issue_data.get("claim_id", 0),
                    why=issue_data.get("why", ""),
                    citations=issue_data.get("citations", []),
                    suggested_edit=issue_data.get("suggested_edit"),
                )
                issues.append(issue)
            except Exception as e:
                logger.warning(f"Issue 파싱 실패: {e}")
                continue

        # additional_sources 파싱 (Perplexity citations 포함)
        additional_sources = data.get("additional_sources", [])

        # Perplexity citations도 추가
        for url in citations:
            if not any(s.get("url") == url for s in additional_sources):
                additional_sources.append(
                    {
                        "title": "",
                        "url": url,
                        "snippet": "",
                    }
                )

        return VerificationResult(
            verdict=verdict,
            issues=issues,
            suggested_counter_queries=data.get("suggested_counter_queries", []),
            confidence=data.get("confidence", 0.5),
            elapsed_ms=elapsed_ms,
            perplexity_model=self.model,
            additional_sources=additional_sources,
        )

    def challenge_claim(self, claim: str) -> Dict:
        """단일 claim 반박 시도

        Args:
            claim: 검증할 주장

        Returns:
            {"verified": bool, "counter_evidence": str, "citations": List[str]}
        """
        prompt = f"""다음 주장을 최신 정보로 검증하고,
반박할 수 있는 근거가 있다면 제시하세요:

주장: {claim}

JSON 응답:
{{
    "verified": true/false,
    "confidence": 0.0-1.0,
    "counter_evidence": "반박 근거 (있다면)",
    "updated_info": "더 최신 정보 (있다면)",
    "citations": ["출처 URL"]
}}"""

        try:
            response = self.client.query(prompt)
            content = response.get("content", "")

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                data["citations"] = data.get("citations", []) + response.get(
                    "citations", []
                )
                return data

        except Exception as e:
            logger.error(f"Claim 검증 실패: {e}")

        return {
            "verified": True,
            "confidence": 0.5,
            "counter_evidence": None,
            "updated_info": None,
            "citations": [],
        }


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adversary Agent 테스트")
    parser.add_argument("--claim", type=str, help="검증할 단일 claim")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.claim:
        agent = AdversaryAgent()
        result = agent.challenge_claim(args.claim)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("사용법: python adversary.py --claim '검증할 주장'")
