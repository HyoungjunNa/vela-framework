"""Confidence Gate - 품질 게이트 모듈"""

import json
import logging
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    passed: bool
    overall_confidence: int
    section_scores: Dict[str, int]
    failed_sections: list
    recommendation: str
    risk_level: str
    reason: str


class ConfidenceGate:
    """Confidence 기반 품질 게이트"""

    def __init__(self, min_overall: int = 90, min_section: int = 80):
        self.min_overall = min_overall
        self.min_section = min_section

    def parse_confidence_json(self, content: str) -> Optional[Dict]:
        """응답에서 JSON 파싱"""
        # ```json ... ``` 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from code block: {e}")

        # 직접 JSON 파싱 시도
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse content as JSON: {e}")

        # [Confidence: XX%] 패턴 추출
        confidence_matches = re.findall(
            r"\[(?:Overall )?Confidence:\s*(\d+)%?\]", content, re.IGNORECASE
        )
        if confidence_matches:
            return {"overall_confidence": int(confidence_matches[-1])}

        return None

    def evaluate(self, content: str) -> GateResult:
        """게이트 평가"""
        data = self.parse_confidence_json(content)

        if not data:
            return GateResult(
                passed=False,
                overall_confidence=0,
                section_scores={},
                failed_sections=[],
                recommendation="N/A",
                risk_level="N/A",
                reason="Confidence 데이터 파싱 실패",
            )

        overall = data.get("overall_confidence", 0)
        sections = data.get("sections", {})
        recommendation = data.get("recommendation", "N/A")
        risk_level = data.get("risk_level", "N/A")

        # 섹션별 점수 추출
        section_scores = {}
        failed_sections = []

        for name, sec_data in sections.items():
            if isinstance(sec_data, dict):
                score = sec_data.get("confidence", 0)
            else:
                score = 0
            section_scores[name] = score

            if score < self.min_section:
                failed_sections.append(name + " (" + str(score) + "%)")

        # 게이트 통과 여부
        passed = overall >= self.min_overall and len(failed_sections) == 0

        if not passed:
            if overall < self.min_overall:
                reason = (
                    "Overall confidence "
                    + str(overall)
                    + "% < "
                    + str(self.min_overall)
                    + "%"
                )
            else:
                reason = "Low confidence sections: " + ", ".join(failed_sections)
        else:
            reason = "PASSED - All criteria met"

        return GateResult(
            passed=passed,
            overall_confidence=overall,
            section_scores=section_scores,
            failed_sections=failed_sections,
            recommendation=recommendation,
            risk_level=risk_level,
            reason=reason,
        )

    def format_result(self, result: GateResult) -> str:
        """결과 포맷팅"""
        status = "✅ PASS" if result.passed else "❌ REJECT"
        lines = [
            "=" * 50,
            "Quality Gate: " + status,
            "=" * 50,
            "Overall Confidence: " + str(result.overall_confidence) + "%",
            "Min Required: " + str(self.min_overall) + "%",
            "",
            "Section Scores:",
        ]

        for name, score in result.section_scores.items():
            mark = "✓" if score >= self.min_section else "✗"
            lines.append("  " + mark + " " + name + ": " + str(score) + "%")

        lines.extend(
            [
                "",
                "Recommendation: " + result.recommendation,
                "Risk Level: " + result.risk_level,
                "",
                "Result: " + result.reason,
                "=" * 50,
            ]
        )

        return "\n".join(lines)


# 테스트
if __name__ == "__main__":
    gate = ConfidenceGate(min_overall=80, min_section=60)

    # 테스트 케이스 1: 통과
    test1 = """```json
{
  "sections": {
    "price_analysis": {"confidence": 85},
    "financial_analysis": {"confidence": 80},
    "news_sentiment": {"confidence": 75},
    "investor_flow": {"confidence": 82}
  },
  "overall_confidence": 81,
  "recommendation": "매수",
  "risk_level": "보통"
}
```"""

    # 테스트 케이스 2: 실패 (overall 낮음)
    test2 = """```json
{
  "sections": {
    "price_analysis": {"confidence": 85},
    "financial_analysis": {"confidence": 75}
  },
  "overall_confidence": 78,
  "recommendation": "관망",
  "risk_level": "보통"
}
```"""

    # 테스트 케이스 3: 실패 (섹션 낮음)
    test3 = """```json
{
  "sections": {
    "price_analysis": {"confidence": 85},
    "financial_analysis": {"confidence": 55}
  },
  "overall_confidence": 82,
  "recommendation": "매도",
  "risk_level": "높음"
}
```"""

    print("Test 1: Should PASS")
    r1 = gate.evaluate(test1)
    print(gate.format_result(r1))

    print("\nTest 2: Should REJECT (overall < 80)")
    r2 = gate.evaluate(test2)
    print(gate.format_result(r2))

    print("\nTest 3: Should REJECT (section < 60)")
    r3 = gate.evaluate(test3)
    print(gate.format_result(r3))
