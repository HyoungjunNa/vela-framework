"""CoT Research System - Pydantic 스키마 정의

구조화된 JSON 출력을 위한 데이터 모델
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# 유틸리티 함수
# =============================================================================


def generate_source_id(url: str) -> str:
    """URL 기반 안정적 source_id 생성

    동일 URL은 항상 동일 ID → trajectory/claim-evidence에서 일관성 보장

    Args:
        url: 소스 URL

    Returns:
        sha1(url)[:10] 형식의 10자 ID
    """
    return hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


class ActionType(str, Enum):
    """추론 액션 타입"""

    SEARCH = "search"  # 추가 검색 필요
    ANALYZE = "analyze"  # 수집된 데이터 분석
    CONCLUDE = "conclude"  # 결론 도출


class SourceType(str, Enum):
    """소스 타입"""

    NEWS = "news"  # 뉴스 기사
    REPORT = "report"  # 증권사 리포트
    PDF = "pdf"  # PDF 문서
    WEB = "web"  # 일반 웹페이지
    ARCHIVE = "archive"  # BM25 과거 뉴스
    PRICE = "price"  # 실시간 시세 (pykis/pykiwoom)
    CHART = "chart"  # 차트 데이터 (일봉/분봉)
    INVESTOR = "investor"  # 투자자별 매매동향
    RANKING = "ranking"  # 거래량/등락률 랭킹
    SECTOR = "sector"  # 업종 분석
    FUNDAMENTAL = "fundamental"  # 재무정보 (FnGuide)
    KEYWORD_EFFECT = "keyword_effect"  # 키워드 주가효과 (BM25 + time-weighted)
    CONSENSUS = "consensus"  # 증권사 목표주가 컨센서스 (INT-354)


# =============================================================================
# Adversary Agent 검증 스키마
# =============================================================================


class VerificationVerdict(str, Enum):
    """검증 판정 결과"""

    ACCEPT = "accept"  # 검증 통과
    REVISE = "revise"  # 수정 필요
    NEED_MORE_SEARCH = "need_more_search"  # 추가 검색 필요


class IssueType(str, Enum):
    """검증 이슈 타입"""

    UNSUPPORTED_CLAIM = "unsupported_claim"  # 근거 없는 주장
    CONTRADICTION = "contradiction"  # 모순된 정보
    STALE_INFO = "stale_info"  # 오래된 정보
    MISSING_KEY_FACT = "missing_key_fact"  # 핵심 사실 누락


class IssueSeverity(str, Enum):
    """이슈 심각도"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationIssue(BaseModel):
    """검증에서 발견된 개별 이슈"""

    type: IssueType = Field(..., description="이슈 타입")
    severity: IssueSeverity = Field(..., description="심각도")
    claim_id: int = Field(..., description="문제가 있는 claim 인덱스")
    why: str = Field(..., description="이슈 발생 이유")
    citations: List[str] = Field(
        default_factory=list,
        description="Perplexity가 제공한 반박/검증 URL",
    )
    suggested_edit: Optional[str] = Field(
        None,
        description="수정 제안 (있는 경우)",
    )


class VerificationResult(BaseModel):
    """Adversary Agent 검증 결과

    Perplexity API를 통한 사실 검증 및 재생성 트리거
    """

    verdict: VerificationVerdict = Field(
        ...,
        description="최종 판정 (accept/revise/need_more_search)",
    )
    issues: List[VerificationIssue] = Field(
        default_factory=list,
        description="발견된 이슈 목록",
    )
    suggested_counter_queries: List[str] = Field(
        default_factory=list,
        description="추가 검색 쿼리 제안 (need_more_search 시)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="검증 신뢰도",
    )

    # 메타데이터
    verified_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="검증 시간",
    )
    perplexity_model: str = Field(
        default="sonar",
        description="사용된 Perplexity 모델",
    )
    elapsed_ms: int = Field(default=0, description="검증 소요 시간 (ms)")

    # Perplexity 제공 추가 소스
    additional_sources: List[Dict[str, str]] = Field(
        default_factory=list,
        description="검증 중 발견된 추가 URL [{title, url, snippet}]",
    )

    class Config:
        use_enum_values = True


class ReasoningStep(BaseModel):
    """단일 추론 스텝

    CoT (Chain-of-Thought) 패턴의 각 단계를 표현
    """

    step_number: int = Field(..., description="추론 단계 번호 (1부터 시작)")
    thought: str = Field(..., description="현재 상황에 대한 사고 과정")
    action: ActionType = Field(..., description="수행할 액션 타입")
    query: Optional[str] = Field(None, description="검색 쿼리 (action=search일 때)")
    observation: str = Field(default="", description="액션 실행 결과 관찰")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="현재 단계의 신뢰도 (0.0~1.0)",
    )
    sources_found: int = Field(default=0, description="이 단계에서 발견한 소스 수")
    elapsed_ms: int = Field(default=0, description="이 단계 소요 시간 (밀리초)")

    class Config:
        use_enum_values = True


class Source(BaseModel):
    """검색 소스 (뉴스, 리포트, PDF 등)"""

    source_id: str = Field(default="", description="안정적 소스 ID (sha1(url)[:10])")
    title: str = Field(..., description="소스 제목")
    url: str = Field(..., description="소스 URL")
    source_type: SourceType = Field(..., description="소스 타입")
    date: str = Field(default="", description="발행일 (YYYY-MM-DD)")
    snippet: str = Field(default="", description="요약/발췌문 (최대 500자)")
    content: Optional[str] = Field(
        None, description="전체 콘텐츠 (extract_content=True일 때)"
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="쿼리 관련성 점수",
    )
    securities_firm: Optional[str] = Field(
        None, description="증권사명 (source_type=report일 때)"
    )
    target_price: Optional[int] = Field(
        None, description="목표주가 (리포트에서 추출된 경우)"
    )
    rating: Optional[str] = Field(None, description="투자의견 (매수/중립/매도 등)")

    @model_validator(mode="after")
    def generate_stable_source_id(self) -> "Source":
        """URL 기반 안정적 source_id 자동 생성"""
        if not self.source_id and self.url:
            object.__setattr__(self, "source_id", generate_source_id(self.url))
        return self

    class Config:
        use_enum_values = True


class ResearchOptions(BaseModel):
    """리서치 옵션"""

    max_iterations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="최대 추론 반복 횟수",
    )
    include_reports: bool = Field(
        default=True, description="증권사 리포트 검색 포함 여부"
    )
    extract_content: bool = Field(default=True, description="전체 콘텐츠 추출 여부")
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="조기 종료 신뢰도 임계값",
    )
    max_sources: int = Field(
        default=20,
        ge=1,
        le=50,
        description="최대 소스 수",
    )
    news_days_back: int = Field(
        default=30,
        ge=1,
        le=365,
        description="뉴스 검색 기간 (일)",
    )
    timeout_seconds: int = Field(
        default=180,
        ge=30,
        le=600,
        description="전체 리서치 타임아웃 (초)",
    )

    # Adversary Agent 검증 옵션
    enable_verification: bool = Field(
        default=False,
        description="Adversary Agent 검증 활성화 (Perplexity API 필요)",
    )
    verification_model: str = Field(
        default="sonar",
        description="검증에 사용할 Perplexity 모델 (sonar, sonar-pro)",
    )
    auto_revise: bool = Field(
        default=False,
        description="검증 실패 시 자동 재생성 (최대 1회)",
    )


class ToolCallRecord(BaseModel):
    """도구 호출 기록 (학습 데이터용)"""

    tool_name: str = Field(..., description="도구 이름")
    input_params: Dict[str, Any] = Field(
        default_factory=dict, description="입력 파라미터"
    )
    output_summary: str = Field(default="", description="출력 요약")
    success: bool = Field(default=True, description="성공 여부")
    elapsed_ms: int = Field(default=0, description="소요 시간 (밀리초)")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="호출 시간",
    )


# =============================================================================
# 학습용 Trajectory & Claim-Evidence (Policy + Writer 데이터)
# =============================================================================


class TodoPriority(str, Enum):
    """TODO 우선순위"""

    CRITICAL = "critical"  # 필수 - 완료 전 conclude 불가
    HIGH = "high"  # 중요 - 가능하면 완료
    MEDIUM = "medium"  # 보통
    LOW = "low"  # 시간 있으면


class ResearchTodoItem(BaseModel):
    """리서치 TODO 아이템

    각 추론 단계에서 LLM이 생성하고 추적하는 작업 항목
    """

    id: str = Field(..., description="TODO ID (t1, t2, ...)")
    task: str = Field(..., description="수행할 작업 설명")
    priority: TodoPriority = Field(
        default=TodoPriority.MEDIUM,
        description="우선순위",
    )
    status: str = Field(
        default="pending",
        description="상태: pending, in_progress, done, skipped",
    )
    created_at_step: int = Field(default=1, description="생성된 스텝 번호")
    completed_at_step: Optional[int] = Field(None, description="완료된 스텝 번호")
    search_query: Optional[str] = Field(
        None,
        description="이 TODO를 위한 검색 쿼리 (있으면)",
    )
    result_summary: Optional[str] = Field(
        None,
        description="완료 시 결과 요약",
    )

    class Config:
        use_enum_values = True


class TrajectoryState(BaseModel):
    """Trajectory 상태 (정규화된 요약)"""

    sources_count: int = Field(default=0, description="현재까지 수집된 소스 수")
    unique_sources_count: int = Field(
        default=0, description="중복 제거 후 고유 소스 수"
    )
    search_queries_all: List[str] = Field(
        default_factory=list,
        description="전체 검색 쿼리 기록 (중복 방지용)",
    )
    search_queries_recent: List[str] = Field(
        default_factory=list,
        description="최근 검색 쿼리 (최대 3개)",
    )
    consecutive_search_count: int = Field(
        default=0,
        description="연속 search 횟수 (analyze 강제 트리거용)",
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="아직 답변되지 않은 질문",
    )
    hypotheses: List[str] = Field(
        default_factory=list,
        description="현재까지의 가설",
    )
    todo_list: List[ResearchTodoItem] = Field(
        default_factory=list,
        description="리서치 TODO 리스트",
    )
    completed_todo_ids: List[str] = Field(
        default_factory=list,
        description="완료된 TODO ID 목록",
    )


class TrajectoryObservation(BaseModel):
    """Trajectory 관찰 결과"""

    added_source_ids: List[str] = Field(
        default_factory=list,
        description="이 스텝에서 추가된 소스 ID",
    )
    top_snippets: List[Dict[str, str]] = Field(
        default_factory=list,
        description="상위 스니펫 [{source_id, snippet}]",
    )


class TrajectoryStep(BaseModel):
    """단일 Trajectory 스텝 (Policy 학습용)

    pre_state → action + action_input → post_state 매핑 학습에 사용
    """

    step: int = Field(..., description="스텝 번호")
    pre_state: TrajectoryState = Field(
        default_factory=TrajectoryState,
        description="액션 실행 전 상태",
    )
    action: str = Field(..., description="수행한 액션 (search/analyze/conclude)")
    action_input: Optional[str] = Field(
        None,
        description="액션 입력 (검색 쿼리 등)",
    )
    observation: TrajectoryObservation = Field(
        default_factory=TrajectoryObservation,
        description="관찰 결과",
    )
    post_state: TrajectoryState = Field(
        default_factory=TrajectoryState,
        description="액션 실행 후 상태",
    )
    rationale: str = Field(
        default="",
        description="결정 근거 (1-2문장)",
    )
    confidence: float = Field(default=0.5, description="신뢰도")


class ClaimEvidence(BaseModel):
    """단일 Claim-Evidence 매핑 (Writer 학습용)

    sources/evidence → claim/conclusion 매핑 학습에 사용
    """

    claim: str = Field(..., description="결론/주장")
    evidence: List[Dict[str, str]] = Field(
        default_factory=list,
        description="근거 [{source_id, support}]",
    )
    risk: Optional[str] = Field(
        None,
        description="리스크/불확실성 (추정 포함 여부 등)",
    )


class ResearchMetadata(BaseModel):
    """리서치 메타데이터 (학습 데이터 생성용)"""

    # 기본 정보
    started_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="시작 시간 (ISO 8601)",
    )
    completed_at: Optional[str] = Field(None, description="완료 시간 (ISO 8601)")
    iterations: int = Field(default=0, description="총 반복 횟수")
    sources_count: int = Field(default=0, description="총 소스 수")
    elapsed_seconds: float = Field(default=0.0, description="총 소요 시간 (초)")

    # LLM 호출 정보
    llm_calls: int = Field(default=0, description="LLM 호출 횟수")
    tokens_used: int = Field(default=0, description="사용된 토큰 수 (추정)")
    model_name: str = Field(default="", description="사용된 모델명")

    # 도구 호출 기록 (학습용)
    tool_calls: List[ToolCallRecord] = Field(
        default_factory=list,
        description="도구 호출 기록 리스트",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="실행된 검색 쿼리 리스트",
    )

    # 소스별 통계
    sources_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="소스 타입별 개수 (news: 10, report: 5 등)",
    )

    # 추론 통계
    avg_confidence: float = Field(default=0.0, description="평균 신뢰도")
    final_confidence: float = Field(default=0.0, description="최종 신뢰도")
    action_sequence: List[str] = Field(
        default_factory=list,
        description="액션 시퀀스 (search → search → conclude)",
    )

    # 에러 정보
    error: Optional[str] = Field(None, description="에러 발생 시 메시지")
    warnings: List[str] = Field(default_factory=list, description="경고 메시지 리스트")

    # 출력 파일 경로
    report_path: Optional[str] = Field(None, description="생성된 리포트 파일 경로")
    reasoning_trace_path: Optional[str] = Field(
        None, description="추론 트레이스 JSON 경로"
    )

    # Reasoning Trace 로그 경로
    reasoning_trace_path: Optional[str] = Field(
        None, description="저장된 reasoning trace JSON 파일 경로"
    )

    # ==========================================================================
    # 학습용 데이터 (Policy + Writer)
    # ==========================================================================

    # Policy 학습용: state → action + action_input 매핑
    trajectory: List[TrajectoryStep] = Field(
        default_factory=list,
        description="Step별 Trajectory 데이터 (Policy 학습용)",
    )

    # Writer 학습용: sources/evidence → claim/conclusion 매핑
    claim_evidence_map: List[ClaimEvidence] = Field(
        default_factory=list,
        description="Claim-Evidence 매핑 (Writer 학습용)",
    )

    # Adversary Agent 검증 결과
    verification: Optional["VerificationResult"] = Field(
        None,
        description="Adversary Agent 검증 결과 (Perplexity)",
    )


class ResearchResult(BaseModel):
    """리서치 결과 (최종 출력)

    구조화된 JSON으로 직렬화 가능
    """

    # 입력
    query: str = Field(..., description="원본 리서치 쿼리")
    options: ResearchOptions = Field(
        default_factory=ResearchOptions, description="적용된 옵션"
    )

    # 추론 과정
    reasoning_trace: List[ReasoningStep] = Field(
        default_factory=list, description="CoT 추론 과정"
    )

    # 수집된 소스
    sources: List[Source] = Field(default_factory=list, description="수집된 소스 목록")

    # 결론
    conclusion: str = Field(default="", description="최종 결론 (마크다운)")
    key_findings: List[str] = Field(
        default_factory=list, description="핵심 발견사항 (불릿포인트)"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="최종 신뢰도",
    )

    # 메타데이터
    metadata: ResearchMetadata = Field(
        default_factory=ResearchMetadata, description="리서치 메타데이터"
    )

    # 디버깅/학습용 필드
    trajectory: List["TrajectoryStep"] = Field(
        default_factory=list,
        description="Policy 학습용 Trajectory (pre_state, action, post_state)",
    )
    data_gaps: List[str] = Field(
        default_factory=list,
        description="미제공 데이터 목록 (밸류에이션 등 소스에서 미확인된 항목)",
    )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """JSON 문자열로 변환"""
        return self.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        """마크다운 리포트로 변환 (STONKS EOD 형식)

        synthesis 단계에서 이미 EOD 리포트 구조로 생성되므로
        conclusion을 직접 출력하고, 메타데이터는 하단에 접힌 형식으로 표시.
        """
        # 결론이 이미 EOD 리포트 형식 — 그대로 출력
        md = self.conclusion.strip() + "\n"

        # 하단 메타데이터 (Gradio accordion용)
        md += "\n---\n"
        md += f"*신뢰도: {self.confidence:.0%} · 소스: {len(self.sources)}개 · "
        md += f"소요: {self.metadata.elapsed_seconds:.0f}초*\n"

        return md


# ============================================================================
# 요청/응답 스키마 (API용)
# ============================================================================


class ResearchRequest(BaseModel):
    """리서치 요청 (API 입력)"""

    query: str = Field(..., min_length=2, max_length=500, description="리서치 쿼리")
    stock_code: Optional[str] = Field(None, description="종목코드 (선택)")
    stock_name: Optional[str] = Field(None, description="종목명 (선택)")
    options: ResearchOptions = Field(
        default_factory=ResearchOptions, description="리서치 옵션"
    )


class ResearchResponse(BaseModel):
    """리서치 응답 (API 출력)"""

    success: bool = Field(..., description="성공 여부")
    result: Optional[ResearchResult] = Field(None, description="리서치 결과")
    error: Optional[str] = Field(None, description="에러 메시지")


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    # 샘플 데이터 생성
    step1 = ReasoningStep(
        step_number=1,
        thought="HBM 시장 전망을 파악하려면 최근 뉴스와 증권사 리포트가 필요합니다.",
        action=ActionType.SEARCH,
        query="SK하이닉스 HBM 수주",
        observation="5개 뉴스 발견: NVIDIA 공급계약 관련 기사 다수",
        confidence=0.6,
        sources_found=5,
    )

    source1 = Source(
        title="SK하이닉스, 엔비디아 HBM3e 독점 공급",
        url="https://example.com/news/123",
        source_type=SourceType.NEWS,
        date="2026-01-02",
        snippet="SK하이닉스가 엔비디아에 HBM3e 메모리를 독점 공급한다고 발표...",
        relevance_score=0.92,
    )

    result = ResearchResult(
        query="SK하이닉스 HBM 시장 전망",
        reasoning_trace=[step1],
        sources=[source1],
        conclusion="SK하이닉스는 HBM3e 시장에서 독점적 지위를 확보하며...",
        key_findings=[
            "HBM3e 시장 점유율 50% 이상",
            "NVIDIA 독점 공급 계약 체결",
            "2026년 매출 30% 성장 전망",
        ],
        confidence=0.87,
    )

    print("=== JSON 출력 ===")
    print(result.to_json())

    print("\n=== 마크다운 출력 ===")
    print(result.to_markdown())
