"""FactExtractor - 사실 기반 데이터 추출기 (Perplexity 스타일)

Created: 2026-01-03
Purpose: 뉴스/웹에서 구조화된 사실 데이터 추출

패턴: 검색 → 크롤링 → LLM 구조화 추출 → 검증 → 출처+신뢰도 반환

사용 예시:
    # 스키마 정의
    schema = FactSchema(
        name="earnings",
        description="기업 실적 발표",
        fields={
            "revenue": {"type": "int", "description": "매출액 (억원)"},
            "operating_profit": {"type": "int", "description": "영업이익 (억원)"},
            "net_profit": {"type": "int", "description": "순이익 (억원)"},
            "yoy_growth": {"type": "float", "description": "전년 대비 성장률 (%)"},
        },
        required_fields=["revenue", "operating_profit"],
    )

    # 추출기 생성 및 실행
    extractor = FactExtractor(schema)
    results = extractor.extract(
        query="삼성전자 4분기 실적",
        entity_name="삼성전자",
    )
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# =============================================================================
# 스키마 정의
# =============================================================================


@dataclass
class FieldSchema:
    """필드 스키마 정의"""

    name: str
    type: str  # int, float, str, bool, list
    description: str
    required: bool = False
    enum: Optional[List[str]] = None  # 허용 값 목록
    min_value: Optional[float] = None  # 숫자 최소값
    max_value: Optional[float] = None  # 숫자 최대값


@dataclass
class FactSchema:
    """사실 추출 스키마"""

    name: str  # 스키마 이름 (예: "target_price", "earnings")
    description: str  # 스키마 설명
    fields: Dict[str, FieldSchema]  # 필드 정의
    entity_field: str = "entity_name"  # 엔티티 필드명 (종목명 등)
    validation_prompt: str = ""  # 추가 검증 프롬프트

    @classmethod
    def from_dict(cls, data: Dict) -> "FactSchema":
        """딕셔너리에서 스키마 생성"""
        fields = {}
        for name, field_def in data.get("fields", {}).items():
            if isinstance(field_def, dict):
                fields[name] = FieldSchema(
                    name=name,
                    type=field_def.get("type", "str"),
                    description=field_def.get("description", ""),
                    required=field_def.get("required", False),
                    enum=field_def.get("enum"),
                    min_value=field_def.get("min_value"),
                    max_value=field_def.get("max_value"),
                )
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            fields=fields,
            entity_field=data.get("entity_field", "entity_name"),
            validation_prompt=data.get("validation_prompt", ""),
        )

    def to_json_schema(self) -> str:
        """LLM 프롬프트용 JSON 스키마 생성"""
        schema = {}
        for name, field in self.fields.items():
            type_hint = field.type
            if field.enum:
                type_hint = f'"{"|".join(field.enum)}"'
            schema[name] = f"{type_hint} // {field.description}"

        return json.dumps(schema, ensure_ascii=False, indent=2)


# =============================================================================
# 추출 결과
# =============================================================================


@dataclass
class FactRecord:
    """추출된 사실 레코드"""

    schema_name: str  # 스키마 이름
    entity_name: str  # 엔티티명 (종목명 등)
    data: Dict[str, Any]  # 추출된 데이터
    source_url: str  # 출처 URL
    source_title: str = ""  # 출처 제목
    source_date: Optional[str] = None  # 출처 날짜
    confidence: float = 0.0  # 신뢰도 (0~1)
    llm_reason: str = ""  # LLM 판단 근거
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_content: str = ""  # 원본 콘텐츠 (디버깅용)

    def to_dict(self) -> Dict:
        return {
            "schema_name": self.schema_name,
            "entity_name": self.entity_name,
            "data": self.data,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "source_date": self.source_date,
            "confidence": self.confidence,
            "llm_reason": self.llm_reason,
            "extracted_at": self.extracted_at,
        }

    def __repr__(self) -> str:
        return (
            f"FactRecord({self.schema_name}: {self.entity_name}, "
            f"confidence={self.confidence:.0%}, url={self.source_url[:50]}...)"
        )


# =============================================================================
# 가드레일 (검증기)
# =============================================================================


class Guardrail(ABC):
    """가드레일 추상 클래스"""

    @abstractmethod
    def validate(self, record: FactRecord) -> Tuple[bool, str]:
        """레코드 검증

        Returns:
            (통과 여부, 실패 사유)
        """
        pass


class RangeGuardrail(Guardrail):
    """숫자 범위 가드레일"""

    def __init__(
        self,
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        reference_getter: Optional[Callable[[], Optional[float]]] = None,
        min_ratio: float = 0.0,
        max_ratio: float = float("inf"),
    ):
        """
        Args:
            field_name: 검증할 필드명
            min_value: 절대 최소값
            max_value: 절대 최대값
            reference_getter: 참조값 조회 함수 (예: 현재가 조회)
            min_ratio: 참조값 대비 최소 비율
            max_ratio: 참조값 대비 최대 비율
        """
        self.field_name = field_name
        self.min_value = min_value
        self.max_value = max_value
        self.reference_getter = reference_getter
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def validate(self, record: FactRecord) -> Tuple[bool, str]:
        value = record.data.get(self.field_name)
        if value is None:
            return True, ""  # 값 없으면 통과

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False, f"{self.field_name} 값이 숫자가 아님: {value}"

        # 절대 범위 검증
        if self.min_value is not None and value < self.min_value:
            return False, f"{self.field_name}={value} < 최소값 {self.min_value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"{self.field_name}={value} > 최대값 {self.max_value}"

        # 참조값 대비 비율 검증
        if self.reference_getter:
            ref_value = self.reference_getter()
            if ref_value:
                min_allowed = ref_value * self.min_ratio
                max_allowed = ref_value * self.max_ratio
                if value < min_allowed:
                    return (
                        False,
                        f"{self.field_name}={value} < 참조값({ref_value}) * {self.min_ratio}",
                    )
                if value > max_allowed:
                    return (
                        False,
                        f"{self.field_name}={value} > 참조값({ref_value}) * {self.max_ratio}",
                    )

        return True, ""


class DateGuardrail(Guardrail):
    """날짜 범위 가드레일"""

    def __init__(self, max_days_old: int = 30):
        self.max_days_old = max_days_old

    def validate(self, record: FactRecord) -> Tuple[bool, str]:
        if not record.source_date:
            return True, ""

        try:
            # 다양한 날짜 형식 파싱
            date_str = record.source_date[:10]
            if "-" in date_str:
                record_date = datetime.strptime(date_str, "%Y-%m-%d")
            elif "." in date_str:
                record_date = datetime.strptime(date_str, "%Y.%m.%d")
            else:
                return True, ""  # 파싱 불가시 통과

            days_old = (datetime.now() - record_date).days
            if days_old > self.max_days_old:
                return False, f"날짜가 {days_old}일 전 (최대 {self.max_days_old}일)"

        except ValueError:
            pass  # 파싱 실패시 통과

        return True, ""


class CompositeGuardrail(Guardrail):
    """복합 가드레일"""

    def __init__(self, guardrails: List[Guardrail]):
        self.guardrails = guardrails

    def validate(self, record: FactRecord) -> Tuple[bool, str]:
        for guardrail in self.guardrails:
            passed, reason = guardrail.validate(record)
            if not passed:
                return False, reason
        return True, ""


# =============================================================================
# LLM 추출기
# =============================================================================


class LLMFactExtractor:
    """LLM 기반 사실 추출기"""

    EXTRACTION_PROMPT_TEMPLATE = """당신은 뉴스/문서에서 사실 정보를 추출하는 전문가입니다.

## 작업
아래 뉴스/문서에서 **{entity_name}**에 대한 {schema_description} 정보를 추출하세요.
다른 대상에 대한 정보는 무시하세요.

## 뉴스/문서
제목: {title}
내용: {content}

## 추출할 정보 (JSON)
{{
    "is_valid": true/false,  // {entity_name}에 대한 정보가 실제로 있는지
    "data": {{
{fields_schema}
    }},
    "confidence": 0.0~1.0,  // 추출 신뢰도
    "reason": "판단 근거 한 줄"
}}

{validation_prompt}

## 주의사항
1. {entity_name} 외 다른 대상의 정보면 is_valid=false
2. 정보가 명확하지 않으면 해당 필드는 null
3. 숫자는 단위 변환하여 정수로 (예: "22만" → 220000)

JSON만 출력하세요:"""

    def __init__(
        self,
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        timeout: int = 30,
    ):
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.model = model
        self.timeout = timeout
        self._client = None
        self._connected = False

    def connect(self) -> bool:
        """vLLM 서버 연결"""
        try:
            from src.report.vllm_client import VLLMClient, VLLMConfig

            config = VLLMConfig(
                host=self.vllm_host,
                port=self.vllm_port,
                model=self.model,
                timeout=self.timeout,
            )
            self._client = VLLMClient(config)
            self._connected = self._client.connect()
            if self._connected:
                logger.info("✅ LLM 추출기 연결 성공")
            return self._connected
        except ImportError:
            logger.warning("VLLMClient import 실패")
            return False
        except Exception as e:
            logger.warning(f"vLLM 연결 실패: {e}")
            return False

    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def extract(
        self,
        schema: FactSchema,
        entity_name: str,
        title: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """LLM으로 사실 추출

        Returns:
            {"is_valid": bool, "data": dict, "confidence": float, "reason": str}
        """
        if not self.is_connected():
            return None

        # 필드 스키마 생성
        fields_lines = []
        for name, field in schema.fields.items():
            type_hint = field.type
            if field.enum:
                type_hint = f'"{"/".join(field.enum)}"'
            required_mark = " (필수)" if field.required else ""
            fields_lines.append(
                f'        "{name}": {type_hint} 또는 null,  // {field.description}{required_mark}'
            )

        prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(
            entity_name=entity_name,
            schema_description=schema.description,
            title=title,
            content=content[:2000],  # 토큰 제한
            fields_schema="\n".join(fields_lines),
            validation_prompt=schema.validation_prompt,
        )

        try:
            response = self._client.chat(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
            )

            if not response:
                return None

            return self._parse_json_response(response)

        except Exception as e:
            logger.warning(f"LLM 추출 실패: {e}")
            return None

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """JSON 응답 파싱"""
        try:
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            return json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            return None


# =============================================================================
# 메인 FactExtractor 클래스
# =============================================================================


class FactExtractor:
    """사실 기반 데이터 추출기 (Perplexity 스타일)

    패턴: 검색 → 크롤링 → LLM 구조화 추출 → 검증 → 출처+신뢰도 반환
    """

    def __init__(
        self,
        schema: FactSchema,
        guardrails: Optional[List[Guardrail]] = None,
        min_confidence: float = 0.5,
        auto_connect: bool = True,
    ):
        """
        Args:
            schema: 추출할 데이터 스키마
            guardrails: 가드레일 목록
            min_confidence: 최소 신뢰도 (이하는 필터링)
            auto_connect: vLLM 자동 연결
        """
        self.schema = schema
        self.guardrails = CompositeGuardrail(guardrails) if guardrails else None
        self.min_confidence = min_confidence

        # 검색기
        self._search_tool = None
        self._init_search_tool()

        # 크롤러
        self._content_extractor = None
        self._init_content_extractor()

        # LLM 추출기
        self._llm_extractor = LLMFactExtractor()
        if auto_connect:
            self._llm_extractor.connect()

    def _init_search_tool(self):
        """검색 도구 초기화"""
        try:
            from .naver_search import NaverSearchTool

            self._search_tool = NaverSearchTool()
        except ImportError:
            try:
                from src.vela_agent.tools.naver_search import NaverSearchTool

                self._search_tool = NaverSearchTool()
            except ImportError:
                logger.warning("NaverSearchTool import 실패")

    def _init_content_extractor(self):
        """콘텐츠 추출기 초기화"""
        try:
            from ..research.content_extractor import ContentExtractor

            self._content_extractor = ContentExtractor()
        except ImportError:
            try:
                from src.vela_agent.research.content_extractor import ContentExtractor

                self._content_extractor = ContentExtractor()
            except ImportError:
                logger.warning("ContentExtractor import 실패")

    def extract(
        self,
        query: str,
        entity_name: str,
        days_back: int = 7,
        max_results: int = 30,
    ) -> List[FactRecord]:
        """사실 데이터 추출

        Args:
            query: 검색 쿼리
            entity_name: 엔티티명 (종목명 등)
            days_back: 검색 기간 (일)
            max_results: 최대 결과 수

        Returns:
            FactRecord 리스트
        """
        logger.info(f"📡 사실 추출 시작: {query} (엔티티: {entity_name})")

        # 1단계: 뉴스 검색
        news_items = self._search_news(query, days_back, max_results)
        if not news_items:
            logger.warning("검색 결과 없음")
            return []

        logger.info(f"🔍 {len(news_items)}개 뉴스 수집")

        # 2단계: 크롤링 + LLM 추출
        records = self._extract_from_items(news_items, entity_name)
        logger.info(f"📊 {len(records)}개 사실 추출")

        # 3단계: 가드레일 적용
        if self.guardrails:
            before_count = len(records)
            records = self._apply_guardrails(records)
            filtered = before_count - len(records)
            if filtered > 0:
                logger.info(f"🚫 가드레일 필터: {filtered}건 제외")

        # 4단계: 최소 신뢰도 필터
        records = [r for r in records if r.confidence >= self.min_confidence]

        logger.info(f"✅ 최종 결과: {len(records)}건")
        return records

    def _search_news(
        self,
        query: str,
        days_back: int,
        max_results: int,
    ) -> List[Dict]:
        """뉴스 검색"""
        if not self._search_tool:
            return []

        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            results = self._search_tool.search_news_for_trading_date(
                query=query,
                trading_date=end_date,
                trading_days_back=days_back,
                max_results=max_results,
            )

            return [
                {
                    "title": r.title,
                    "description": r.description,
                    "url": r.url,
                    "pub_date": r.pub_date,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"뉴스 검색 실패: {e}")
            return []

    def _extract_from_items(
        self,
        items: List[Dict],
        entity_name: str,
    ) -> List[FactRecord]:
        """뉴스 항목에서 사실 추출"""
        records = []
        seen_urls = set()

        for item in items:
            url = item.get("url") or item.get("link")

            # URL 필수
            if not url:
                continue

            # 중복 제거
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = item.get("title", "")

            # 본문 크롤링
            content = None
            content_crawled = False
            if self._content_extractor:
                try:
                    content = self._content_extractor.extract(url)
                    if content:
                        content_crawled = True
                except Exception as e:
                    logger.debug(f"크롤링 실패: {url} - {e}")

            if not content:
                content = item.get("description", "") or item.get("content", "")

            # LLM 추출
            if not self._llm_extractor.is_connected():
                continue

            result = self._llm_extractor.extract(
                schema=self.schema,
                entity_name=entity_name,
                title=title,
                content=content,
            )

            if not result or not result.get("is_valid"):
                continue

            # Confidence 계산
            confidence = self._calculate_confidence(
                llm_confidence=result.get("confidence", 0.5),
                data=result.get("data", {}),
                content_crawled=content_crawled,
            )

            record = FactRecord(
                schema_name=self.schema.name,
                entity_name=entity_name,
                data=result.get("data", {}),
                source_url=url,
                source_title=title,
                source_date=(
                    item.get("pub_date", "")[:10] if item.get("pub_date") else None
                ),
                confidence=confidence,
                llm_reason=result.get("reason", ""),
                raw_content=content[:500] if content else "",
            )

            records.append(record)
            logger.info(f"✅ [{confidence:.0%}] {title[:40]}... → {url[:50]}...")

        return records

    def _calculate_confidence(
        self,
        llm_confidence: float,
        data: Dict,
        content_crawled: bool,
    ) -> float:
        """신뢰도 계산

        LLM 신뢰도 (50%) + 추가 요소 (50%)
        """
        # LLM 기본 점수
        base_score = min(llm_confidence, 1.0) * 0.5

        # 필수 필드 충족 보너스
        bonus = 0.0
        required_count = sum(1 for f in self.schema.fields.values() if f.required)
        if required_count > 0:
            filled_required = sum(
                1
                for name, field in self.schema.fields.items()
                if field.required and data.get(name) is not None
            )
            bonus += 0.25 * (filled_required / required_count)

        # 추가 필드 충족 보너스
        optional_fields = [f for f in self.schema.fields.values() if not f.required]
        if optional_fields:
            filled_optional = sum(
                1
                for name, field in self.schema.fields.items()
                if not field.required and data.get(name) is not None
            )
            bonus += 0.15 * (filled_optional / len(optional_fields))

        # 본문 크롤링 성공 보너스
        if content_crawled:
            bonus += 0.10

        return min(base_score + bonus, 1.0)

    def _apply_guardrails(self, records: List[FactRecord]) -> List[FactRecord]:
        """가드레일 적용"""
        if not self.guardrails:
            return records

        valid_records = []
        for record in records:
            passed, reason = self.guardrails.validate(record)
            if passed:
                valid_records.append(record)
            else:
                logger.debug(f"🚫 가드레일 실패: {reason}")

        return valid_records


# =============================================================================
# 프리셋 스키마
# =============================================================================


class FactSchemaPresets:
    """자주 사용하는 스키마 프리셋"""

    @staticmethod
    def earnings() -> FactSchema:
        """실적 발표 스키마"""
        return FactSchema(
            name="earnings",
            description="기업 실적 발표",
            fields={
                "period": FieldSchema(
                    name="period",
                    type="str",
                    description="실적 기간 (예: 2025년 4분기)",
                    required=True,
                ),
                "revenue": FieldSchema(
                    name="revenue",
                    type="int",
                    description="매출액 (억원)",
                    required=True,
                ),
                "operating_profit": FieldSchema(
                    name="operating_profit",
                    type="int",
                    description="영업이익 (억원)",
                    required=True,
                ),
                "net_profit": FieldSchema(
                    name="net_profit",
                    type="int",
                    description="순이익 (억원)",
                ),
                "yoy_revenue_growth": FieldSchema(
                    name="yoy_revenue_growth",
                    type="float",
                    description="매출 전년비 성장률 (%)",
                ),
                "yoy_profit_growth": FieldSchema(
                    name="yoy_profit_growth",
                    type="float",
                    description="영업이익 전년비 성장률 (%)",
                ),
                "guidance": FieldSchema(
                    name="guidance",
                    type="str",
                    description="향후 전망/가이던스",
                ),
            },
        )

    @staticmethod
    def dividend() -> FactSchema:
        """배당 정보 스키마"""
        return FactSchema(
            name="dividend",
            description="배당 정보",
            fields={
                "dividend_per_share": FieldSchema(
                    name="dividend_per_share",
                    type="int",
                    description="주당 배당금 (원)",
                    required=True,
                ),
                "dividend_yield": FieldSchema(
                    name="dividend_yield",
                    type="float",
                    description="배당수익률 (%)",
                ),
                "record_date": FieldSchema(
                    name="record_date",
                    type="str",
                    description="배당기준일",
                ),
                "payment_date": FieldSchema(
                    name="payment_date",
                    type="str",
                    description="배당지급일",
                ),
                "dividend_type": FieldSchema(
                    name="dividend_type",
                    type="str",
                    description="배당 유형",
                    enum=["현금배당", "주식배당", "중간배당", "결산배당"],
                ),
            },
        )

    @staticmethod
    def ma() -> FactSchema:
        """M&A 정보 스키마"""
        return FactSchema(
            name="ma",
            description="인수합병 정보",
            fields={
                "deal_type": FieldSchema(
                    name="deal_type",
                    type="str",
                    description="거래 유형",
                    enum=["인수", "합병", "지분투자", "매각"],
                    required=True,
                ),
                "target_company": FieldSchema(
                    name="target_company",
                    type="str",
                    description="대상 회사명",
                    required=True,
                ),
                "deal_value": FieldSchema(
                    name="deal_value",
                    type="int",
                    description="거래 금액 (억원)",
                ),
                "stake_percent": FieldSchema(
                    name="stake_percent",
                    type="float",
                    description="지분율 (%)",
                ),
                "expected_close_date": FieldSchema(
                    name="expected_close_date",
                    type="str",
                    description="예상 완료일",
                ),
                "deal_status": FieldSchema(
                    name="deal_status",
                    type="str",
                    description="거래 상태",
                    enum=["검토중", "협상중", "계약체결", "완료", "무산"],
                ),
            },
        )

    @staticmethod
    def executive_change() -> FactSchema:
        """임원 변경 스키마"""
        return FactSchema(
            name="executive_change",
            description="임원 변경 정보",
            fields={
                "change_type": FieldSchema(
                    name="change_type",
                    type="str",
                    description="변경 유형",
                    enum=["신규선임", "퇴임", "사임", "해임", "승진"],
                    required=True,
                ),
                "name": FieldSchema(
                    name="name",
                    type="str",
                    description="임원 이름",
                    required=True,
                ),
                "position": FieldSchema(
                    name="position",
                    type="str",
                    description="직위/직책",
                    required=True,
                ),
                "effective_date": FieldSchema(
                    name="effective_date",
                    type="str",
                    description="발효일",
                ),
                "previous_position": FieldSchema(
                    name="previous_position",
                    type="str",
                    description="이전 직위",
                ),
            },
        )


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=== FactExtractor 테스트 ===\n")

    # 실적 스키마로 테스트
    schema = FactSchemaPresets.earnings()

    extractor = FactExtractor(
        schema=schema,
        guardrails=[DateGuardrail(max_days_old=30)],
        min_confidence=0.5,
    )

    # 삼성전자 실적 추출
    results = extractor.extract(
        query="삼성전자 4분기 실적",
        entity_name="삼성전자",
        days_back=14,
        max_results=20,
    )

    print(f"\n=== 결과: {len(results)}건 ===\n")
    for r in sorted(results, key=lambda x: -x.confidence):
        print(f"• [{r.confidence:.0%}] {r.source_title[:50]}...")
        print(f"  데이터: {r.data}")
        print(f"  URL: {r.source_url}")
        print()
