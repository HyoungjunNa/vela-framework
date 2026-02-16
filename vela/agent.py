# pylint: disable=too-many-lines
"""ResearchAgent - CoT 리서치 메인 오케스트레이터

Chain-of-Thought 패턴으로 리서치 실행:
Think → Search → Analyze → (반복) → Conclude

사용 예시:
    agent = ResearchAgent()
    result = agent.research("SK하이닉스 HBM 시장 전망")
    print(result.to_json())
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .content_extractor import ContentExtractor
from .reasoning import CoTReasoningEngine, TodoReasoningResult
from .schemas import (
    ActionType,
    ClaimEvidence,
    ResearchMetadata,
    ResearchOptions,
    ResearchRequest,
    ResearchResponse,
    ResearchResult,
    ResearchTodoItem,
    ReasoningStep,
    Source,
    TodoPriority,
    ToolCallRecord,
    TrajectoryObservation,
    TrajectoryState,
    TrajectoryStep,
    VerificationResult,
    VerificationVerdict,
    generate_source_id,
)
from .search import ResearchSearchModule

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
PROJECT_ROOT = Path.cwd()


class ResearchAgent:
    """CoT 리서치 에이전트

    Perplexity Sonar Pro 스타일의 반복적 추론 시스템:
    1. Think: 현재 상황 분석
    2. Search: 추가 정보 검색
    3. Analyze: 수집된 데이터 분석
    4. (반복)
    5. Conclude: 최종 결론 도출
    """

    def __init__(
        self,
        llm_backend: str = "runpod",
        extract_content: bool = True,
    ):
        """
        Args:
            llm_backend: LLM 백엔드 ("runpod" | "mlx" | "vllm")
            extract_content: 콘텐츠 본문 추출 활성화
        """
        self.llm_backend = llm_backend
        self.extract_content = extract_content

        # 컴포넌트 초기화
        self._init_components()

    def _init_components(self):
        """컴포넌트 초기화"""
        # LLM 클라이언트
        self.llm = self._init_llm()

        # CoT 추론 엔진
        self.reasoning = CoTReasoningEngine(
            llm_client=self.llm,
            max_iterations=5,
            min_confidence=0.8,
        )

        # 멀티소스 검색
        self.search = ResearchSearchModule(
            enable_naver=True,
            enable_ddg=True,
            max_workers=4,
        )

        # 콘텐츠 추출기
        self.extractor = ContentExtractor(
            timeout=10,
            max_content_length=10000,
        )

        logger.info(f"ResearchAgent 초기화 완료 (backend={self.llm_backend})")

    def _init_llm(self):
        """LLM 클라이언트 초기화"""
        if self.llm_backend == "runpod":
            from .tools.runpod_client import RunPodClient

            return RunPodClient()

        elif self.llm_backend == "mlx":
            from .tools.mlx_client import VELAMLXClient

            return VELAMLXClient()

        elif self.llm_backend == "vllm":
            from .tools.vllm_client import VLLMClient

            return VLLMClient()

        else:
            # 기본값: RunPod
            logger.warning(f"알 수 없는 백엔드 '{self.llm_backend}', RunPod 사용")
            from .tools.runpod_client import RunPodClient

            return RunPodClient()

    def research(
        self,
        query: str,
        options: Optional[ResearchOptions] = None,
        stock_code: Optional[str] = None,
        stock_name: Optional[str] = None,
    ) -> ResearchResult:
        """메인 리서치 실행

        Args:
            query: 리서치 쿼리 (예: "SK하이닉스 HBM 시장 전망")
            options: 리서치 옵션
            stock_code: 종목코드 (선택)
            stock_name: 종목명 (선택)

        Returns:
            ResearchResult (JSON 직렬화 가능)
        """
        start_time = time.time()

        # 기본 옵션 설정
        if options is None:
            options = ResearchOptions()

        # 메타데이터 초기화
        metadata = ResearchMetadata(
            started_at=datetime.now().isoformat(),
            model_name=self.llm_backend,
        )

        # 결과 객체 초기화
        result = ResearchResult(
            query=query,
            options=options,
            metadata=metadata,
        )

        # 도구 호출 기록 리스트 (학습용)
        tool_calls: List[ToolCallRecord] = []
        search_queries_list: List[str] = []

        # Trajectory 기록 리스트 (Policy 학습용)
        trajectory: List[TrajectoryStep] = []

        try:
            # 컨텍스트 초기화 (TODO 리스트 포함)
            context = {
                "sources": [],
                "previous_steps": [],
                "search_queries": [],
                "stock_code": stock_code,
                "stock_name": stock_name,
                "todo_list": [],  # TODO 리스트 (Step 2+에서 사용)
            }

            steps: List[ReasoningStep] = []
            sources: List[Source] = []
            current_todo_list: List[ResearchTodoItem] = []  # 전역 TODO 상태

            # =================================================================
            # CoT 추론 루프 (TODO 기반)
            # =================================================================
            iteration = 0
            consecutive_search_count = 0  # 연속 search 횟수 (loop prevention)
            intermediate_findings: List[str] = []  # analyze에서 생성된 중간 발견
            MAX_CONSECUTIVE_SEARCH = 2  # 연속 search 최대 횟수 (2회 후 analyze 강제)
            last_todo_result: Optional[TodoReasoningResult] = None  # 마지막 TODO 결과

            while iteration < options.max_iterations:
                iteration += 1
                logger.info(f"=== 추론 스텝 {iteration} ===")

                # 1. Think: 다음 액션 결정 (TODO 기반)
                reason_start = time.time()
                todo_result = self.reasoning.reason(
                    query=query,
                    context=context,
                    step_number=iteration,
                )
                reason_elapsed = int((time.time() - reason_start) * 1000)

                # TodoReasoningResult에서 step 추출
                step = todo_result.step
                last_todo_result = todo_result

                # TODO 리스트 업데이트
                if todo_result.todo_list:
                    current_todo_list = todo_result.todo_list
                    context["todo_list"] = current_todo_list

                # TODO 상태 로깅
                if iteration == 1 and current_todo_list:
                    logger.info(f"TODO 리스트 생성: {len(current_todo_list)}개")
                    for todo in current_todo_list:
                        logger.debug(f"  [{todo.priority}] {todo.id}: {todo.task}")

                if todo_result.current_todo_id:
                    logger.info(f"현재 TODO: {todo_result.current_todo_id}")

                # Critical TODO 현황 로깅
                if current_todo_list:
                    critical_done = sum(
                        1
                        for t in current_todo_list
                        if t.priority == TodoPriority.CRITICAL and t.status == "done"
                    )
                    critical_total = sum(
                        1
                        for t in current_todo_list
                        if t.priority == TodoPriority.CRITICAL
                    )
                    logger.info(f"Critical TODO: {critical_done}/{critical_total} 완료")

                # LLM 호출 기록
                tool_calls.append(
                    ToolCallRecord(
                        tool_name="reasoning.reason",
                        input_params={
                            "query": query,
                            "step": iteration,
                            "current_todo": todo_result.current_todo_id,
                        },
                        output_summary=f"action={step.action}, confidence={step.confidence:.0%}",
                        success=True,
                        elapsed_ms=reason_elapsed,
                    )
                )

                logger.info(f"Action: {step.action}, Confidence: {step.confidence:.0%}")

                # ====== pre_state 캡처 (액션 실행 전) ======
                pre_sources = list(sources)  # 복사본
                pre_consecutive_search_count = (
                    consecutive_search_count  # 액션 전 값 보존
                )

                # 2. Act: 액션 실행
                # Loop Prevention: 연속 search가 너무 많으면 analyze 강제
                if (
                    step.action == ActionType.SEARCH
                    and consecutive_search_count >= MAX_CONSECUTIVE_SEARCH
                ):
                    logger.warning(
                        f"연속 search {consecutive_search_count}회 도달 - analyze 강제"
                    )
                    step.action = ActionType.ANALYZE
                    step.observation = (
                        f"연속 search {MAX_CONSECUTIVE_SEARCH}회 제한으로 analyze 전환"
                    )

                if step.action == ActionType.SEARCH and step.query:
                    # vNext: 중복 쿼리 방지
                    if step.query in search_queries_list:
                        logger.warning(f"중복 쿼리 스킵: {step.query}")
                        step.observation = f"중복 쿼리 '{step.query}' - 스킵됨"
                        step.sources_found = 0
                        steps.append(step)
                        context["previous_steps"] = steps
                        continue

                    # 검색 실행
                    search_start = time.time()
                    new_sources = self._execute_search(
                        step.query,
                        options=options,
                        stock_code=stock_code,
                    )
                    search_elapsed = int((time.time() - search_start) * 1000)

                    # 검색 기록
                    search_queries_list.append(step.query)
                    tool_calls.append(
                        ToolCallRecord(
                            tool_name="search.search_all",
                            input_params={"query": step.query},
                            output_summary=f"{len(new_sources)}개 소스 수집",
                            success=len(new_sources) > 0,
                            elapsed_ms=search_elapsed,
                        )
                    )

                    # 콘텐츠 추출 (옵션) - options 값 사용 (self.extract_content 버그 수정)
                    if options.extract_content and new_sources:
                        extract_start = time.time()
                        self._extract_contents(new_sources[:3])
                        extract_elapsed = int((time.time() - extract_start) * 1000)

                        tool_calls.append(
                            ToolCallRecord(
                                tool_name="extractor.extract",
                                input_params={"urls_count": min(3, len(new_sources))},
                                output_summary=f"{min(3, len(new_sources))}개 콘텐츠 추출",
                                success=True,
                                elapsed_ms=extract_elapsed,
                            )
                        )

                    # 결과 기록
                    sources.extend(new_sources)
                    step.sources_found = len(new_sources)
                    step.observation = self._summarize_search_results(new_sources)

                    # 연속 search 횟수 증가
                    consecutive_search_count += 1

                    # 컨텍스트 업데이트
                    context["sources"] = sources
                    context["search_queries"].append(step.query)

                elif step.action == ActionType.ANALYZE:
                    # Analyze: 실제 intermediate_findings 생성
                    consecutive_search_count = 0  # analyze 수행 시 연속 search 리셋

                    if sources:
                        analyze_result = self._run_analyze(
                            query=query,
                            sources=sources,
                            previous_findings=intermediate_findings,
                        )
                        new_findings = analyze_result.get("findings", [])
                        open_questions = analyze_result.get("open_questions", [])
                        what_is_missing = analyze_result.get("what_is_missing", [])

                        intermediate_findings.extend(new_findings)

                        # 컨텍스트에 분석 결과 저장 (Synthesize에서 사용)
                        context["open_questions"] = open_questions
                        context["what_is_missing"] = what_is_missing

                        step.observation = (
                            f"중간 발견 {len(new_findings)}개, "
                            f"미답변 질문 {len(open_questions)}개, "
                            f"미제공 데이터 {len(what_is_missing)}개"
                        )
                        logger.info(f"Analyze 결과: {step.observation}")
                        if what_is_missing:
                            logger.info(f"  미제공: {what_is_missing[:3]}")
                    else:
                        step.observation = "분석할 소스 없음 - 검색 필요"

                elif step.action == ActionType.CONCLUDE:
                    # 결론 단계 - 루프 종료
                    step.observation = "충분한 정보 수집 완료"
                    # NOTE: consecutive_search_count 변경은 post_state에서 자동 계산됨

                    # Trajectory 스텝 기록 (conclude)
                    # NOTE: pre_consecutive_search_count 사용 (액션 전 값)
                    traj_step = self._create_trajectory_step(
                        step_num=iteration,
                        pre_sources=pre_sources,
                        post_sources=sources,
                        all_search_queries=search_queries_list,
                        step=step,
                        new_source_ids=[],
                        rationale="충분한 정보가 수집되어 결론 도출 단계로 전환",
                        consecutive_search_count=pre_consecutive_search_count,  # 액션 전 값
                        pre_todo_list=current_todo_list,
                        post_todo_list=current_todo_list,
                    )
                    trajectory.append(traj_step)

                    steps.append(step)
                    break

                # ====== post_state용 새 소스 ID 계산 (sha1 기반) ======
                new_source_ids = []
                if step.sources_found > 0:
                    # 이번 스텝에서 추가된 소스들의 stable ID
                    for src in sources[-step.sources_found :]:
                        new_source_ids.append(src.source_id)

                # Trajectory 스텝 기록 (search/analyze)
                # NOTE: pre_consecutive_search_count 사용 (액션 전 값)
                # _create_trajectory_step 내부에서 action에 따라 post_state 값 계산
                traj_step = self._create_trajectory_step(
                    step_num=iteration,
                    pre_sources=pre_sources,
                    post_sources=sources,
                    all_search_queries=search_queries_list,
                    step=step,
                    new_source_ids=new_source_ids,
                    rationale=step.thought[:100] if step.thought else "",
                    consecutive_search_count=pre_consecutive_search_count,  # 액션 전 값
                    pre_todo_list=current_todo_list,
                    post_todo_list=current_todo_list,
                )
                trajectory.append(traj_step)

                steps.append(step)
                context["previous_steps"] = steps

                # 계속 여부 판단 (TODO 상태 고려)
                if not self.reasoning.should_continue(
                    steps, step.confidence, todo_result=last_todo_result
                ):
                    logger.info("추론 종료 조건 충족")
                    break

            # =================================================================
            # 최종 결론 합성
            # =================================================================
            # 가드레일: pykis/fnguide 소스에서 확보된 데이터를 자동 추출
            # LLM이 판단하지 않고 시스템이 강제로 추출
            from .search import ResearchSearchModule

            confirmed_data = ResearchSearchModule.extract_confirmed_data(sources)
            logger.info(
                f"확보된 데이터 필드: {confirmed_data.get('provided_fields', [])}"
            )

            synthesis_start = time.time()
            synthesis = self.reasoning.synthesize(
                query=query,
                steps=steps,
                sources=sources[: options.max_sources],
                what_is_missing=context.get(
                    "what_is_missing", []
                ),  # 미제공 데이터 전달
                confirmed_data=confirmed_data,  # 가드레일: 확보된 데이터 강제 주입
            )
            synthesis_elapsed = int((time.time() - synthesis_start) * 1000)

            tool_calls.append(
                ToolCallRecord(
                    tool_name="reasoning.synthesize",
                    input_params={
                        "steps_count": len(steps),
                        "sources_count": len(sources),
                    },
                    output_summary=f"confidence={synthesis.get('confidence', 0):.0%}",
                    success=bool(synthesis.get("conclusion")),
                    elapsed_ms=synthesis_elapsed,
                )
            )

            # 결과 업데이트
            result.reasoning_trace = steps
            result.sources = self._deduplicate_sources(sources)[: options.max_sources]
            result.conclusion = synthesis.get("conclusion", "결론 생성 실패")
            result.key_findings = synthesis.get("key_findings", [])
            result.confidence = synthesis.get("confidence", 0.5)

            # 디버깅/학습용 필드 (직접 접근 가능하도록)
            result.trajectory = trajectory
            # data_gaps에서도 pykis 제공 데이터 필터링
            raw_data_gaps = synthesis.get(
                "data_gaps", context.get("what_is_missing", [])
            )
            result.data_gaps = raw_data_gaps

            # vNext: key_findings 후행 추출 fallback
            if not result.key_findings and result.conclusion:
                logger.info("key_findings 비어있음 - conclusion에서 추출 시도")
                result.key_findings = self._extract_key_findings_from_conclusion(
                    conclusion=result.conclusion,
                    intermediate_findings=intermediate_findings,
                )

            # =================================================================
            # 학습용 메타데이터 계산
            # =================================================================
            elapsed_seconds = time.time() - start_time

            # 소스 타입별 통계
            sources_by_type: Dict[str, int] = {}
            for src in result.sources:
                src_type = str(src.source_type)
                sources_by_type[src_type] = sources_by_type.get(src_type, 0) + 1

            # 평균 신뢰도 계산
            avg_confidence = (
                sum(s.confidence for s in steps) / len(steps) if steps else 0.0
            )

            # 액션 시퀀스
            action_sequence = [str(s.action) for s in steps]

            # =================================================================
            # Claim-Evidence 매핑 생성 (Writer 학습용)
            # =================================================================
            claim_evidence_map = self._create_claim_evidence_map(
                key_findings=result.key_findings,
                sources=result.sources,
                conclusion=result.conclusion,
                stock_name=stock_name,  # 엔티티 mismatch 필터링용
            )

            # 메타데이터 업데이트
            result.metadata.completed_at = datetime.now().isoformat()
            result.metadata.iterations = len(steps)
            result.metadata.sources_count = len(result.sources)
            result.metadata.elapsed_seconds = elapsed_seconds
            result.metadata.llm_calls = len(steps) + 1  # 추론 + 합성
            result.metadata.tool_calls = tool_calls
            result.metadata.search_queries = search_queries_list
            result.metadata.sources_by_type = sources_by_type
            result.metadata.avg_confidence = avg_confidence
            result.metadata.final_confidence = result.confidence
            result.metadata.action_sequence = action_sequence

            # 학습용 데이터 (Policy + Writer)
            result.metadata.trajectory = trajectory
            result.metadata.claim_evidence_map = claim_evidence_map

            # =================================================================
            # Adversary Agent 검증 (선택적)
            # =================================================================
            if options.enable_verification:
                verification = self._run_verification(result, options)
                result.metadata.verification = verification

                # 검증 결과에 따른 로깅
                if verification.verdict == VerificationVerdict.ACCEPT:
                    logger.info(
                        f"✅ 검증 통과 (confidence={verification.confidence:.0%})"
                    )
                elif verification.verdict == VerificationVerdict.REVISE:
                    logger.warning(
                        f"⚠️ 수정 필요: {len(verification.issues)}개 이슈 발견"
                    )
                elif verification.verdict == VerificationVerdict.NEED_MORE_SEARCH:
                    logger.warning(
                        f"🔍 추가 검색 필요: {verification.suggested_counter_queries}"
                    )

            logger.info(
                f"리서치 완료: {len(result.sources)}개 소스, "
                f"{len(steps)}회 반복, {elapsed_seconds:.1f}초 소요"
            )

            # =================================================================
            # Reasoning Trace JSON 자동 저장
            # =================================================================
            try:
                trace_path = self.save_reasoning_trace(result)
                result.metadata.reasoning_trace_path = str(trace_path)
            except Exception as trace_err:
                logger.warning(f"Reasoning trace 저장 실패: {trace_err}")

            # =================================================================
            # 최종 리포트 마크다운 저장
            # =================================================================
            try:
                if result.conclusion:
                    report_dir = Path("output/reports")
                    report_dir.mkdir(parents=True, exist_ok=True)

                    safe_query = "".join(
                        c if c.isalnum() or c in "_ " else "_" for c in query
                    )[:50]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_path = report_dir / f"report_{safe_query}_{timestamp}.md"

                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(f"# {query}\n\n")
                        f.write(
                            f"*생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                        )
                        f.write("---\n\n")
                        f.write(result.conclusion)
                        f.write("\n\n---\n\n")
                        f.write("## 핵심 발견\n\n")
                        for i, finding in enumerate(result.key_findings or [], 1):
                            f.write(f"{i}. {finding}\n")
                        f.write("\n## 메타데이터\n\n")
                        f.write(f"- 소스 수: {len(result.sources)}개\n")
                        f.write(f"- 추론 단계: {len(result.reasoning_trace)}회\n")
                        f.write(f"- 신뢰도: {result.confidence:.0%}\n")

                    logger.info(f"리포트 저장: {report_path}")
                    result.metadata.report_path = str(report_path)
            except Exception as report_err:
                logger.warning(f"리포트 저장 실패: {report_err}")

        except Exception as e:
            logger.error(f"리서치 실패: {e}")
            result.metadata.error = str(e)
            result.metadata.completed_at = datetime.now().isoformat()
            result.metadata.elapsed_seconds = time.time() - start_time

        return result

    def _execute_search(
        self,
        query: str,
        options: ResearchOptions,
        stock_code: Optional[str] = None,
    ) -> List[Source]:
        """검색 실행

        Args:
            query: 검색 쿼리
            options: 리서치 옵션
            stock_code: 종목코드 (재무데이터 조회용)

        Returns:
            Source 리스트
        """
        # 쿼리 정규화: search_news("...") 또는 search("...") 형식 처리
        import re

        func_match = re.search(
            r'(?:search_news|search)\s*\(\s*["\']([^"\']+)["\']', query
        )
        if func_match:
            query = func_match.group(1)
            logger.debug(f"쿼리 정규화: {query}")

        # 종목코드 자동 추출 (없으면 쿼리에서 추출)
        if not stock_code:
            stock_code = self.search.resolve_stock_code(query)
            if stock_code:
                logger.info(f"종목코드 자동 추출: {stock_code}")

        # 활성화된 소스 결정
        sources_to_use = ["naver", "ddg"]

        # 검색 실행
        results = self.search.search_all(
            query=query,
            max_results=10,
            sources=sources_to_use,
            stock_code=stock_code,
        )

        return results

    def _extract_contents(self, sources: List[Source]):
        """소스별 콘텐츠 본문 추출 (in-place)

        Args:
            sources: 추출할 소스 리스트
        """
        for src in sources:
            if src.content:
                continue  # 이미 콘텐츠 있음

            try:
                content = self.extractor.extract(src.url)
                if content:
                    src.content = content[:5000]  # 최대 5000자
                    logger.debug(f"콘텐츠 추출 성공: {src.title[:30]}...")
            except Exception as e:
                logger.warning(f"콘텐츠 추출 실패: {src.url} - {e}")

    def _extract_key_findings_from_conclusion(
        self,
        conclusion: str,
        intermediate_findings: List[str],
    ) -> List[str]:
        """key_findings 후행 추출 (fallback)

        vNext: synthesize에서 key_findings가 비어있을 때 호출

        Args:
            conclusion: 최종 결론
            intermediate_findings: analyze에서 생성된 중간 발견

        Returns:
            추출된 key_findings 리스트
        """
        # 1순위: intermediate_findings 활용
        if intermediate_findings:
            logger.info(
                f"intermediate_findings에서 {len(intermediate_findings)}개 활용"
            )
            return intermediate_findings[:5]

        # 2순위: conclusion에서 LLM으로 추출
        extract_prompt = f"""다음 결론에서 핵심 발견사항 3-5개를 추출하세요.

## 결론
{conclusion}

## 요청
JSON 형식으로 응답하세요:
```json
{{"key_findings": ["발견1", "발견2", "발견3"]}}
```
"""

        try:
            result = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "핵심 포인트를 추출하는 분석가입니다.",
                    },
                    {"role": "user", "content": extract_prompt},
                ],
                max_tokens=500,
                temperature=0.2,
            )

            if result.get("success"):
                import json
                import re

                content = result.get("content", "")
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    findings = parsed.get("key_findings", [])
                    if findings:
                        logger.info(f"conclusion에서 {len(findings)}개 추출 성공")
                        return findings

        except Exception as e:
            logger.warning(f"key_findings 추출 실패: {e}")

        # 3순위: 단순 문장 분리 (최후의 fallback)
        sentences = [s.strip() for s in conclusion.split(".") if len(s.strip()) > 20]
        if sentences:
            logger.info(f"문장 분리로 {min(3, len(sentences))}개 추출")
            return sentences[:3]

        return []

    def _run_analyze(
        self,
        query: str,
        sources: List[Source],
        previous_findings: List[str],
    ) -> Dict:
        """Analyze 단계: 수집된 소스에서 중간 발견(intermediate_findings) 생성

        Args:
            query: 원본 쿼리
            sources: 수집된 소스
            previous_findings: 이전에 생성된 발견사항

        Returns:
            {"findings": List[str], "open_questions": List[str]}
        """
        # 소스 요약 생성
        source_summaries = []
        for i, src in enumerate(sources[:10], 1):
            summary = f"{i}. [{src.source_type}] {src.title[:60]}"
            if src.snippet:
                summary += f"\n   {src.snippet[:150]}..."
            source_summaries.append(summary)

        analyze_prompt = f"""## 리서치 쿼리
{query}

## 수집된 소스 ({len(sources)}개)
{chr(10).join(source_summaries)}

## 이전 발견사항
{chr(10).join(f'- {f}' for f in previous_findings) if previous_findings else '없음'}

## 분석 요청
위 소스를 **팩트 중심으로** 분석하여 다음 JSON 형식으로 응답하세요:

```json
{{
    "findings": [
        "구체적 수치가 있는 사실 5~8개"
    ],
    "open_questions": [
        "아직 답변되지 않은 질문"
    ],
    "what_is_missing": [
        "소스에서 확인 불가한 재무수치만 나열 (빈 배열 가능)"
    ]
}}
```

## 중요 규칙
1. findings는 **5~8개**, 반드시 소스에 기반한 구체적 사실만
2. 소스에 없는 수치는 절대 추측하지 말 것
3. what_is_missing 판단 기준:
   - [재무지표] 소스에 PER/PBR/EPS/BPS가 있으면 → 빈 배열 또는 누락된 것만
   - 소스에 밸류에이션 데이터가 없으면 → "PER/PBR: 미제공" 추가
   - 실제 소스를 확인하고 판단할 것 (예시 값 그대로 복사 금지)
4. 투기적 예측("강세 전망", "상승 예상") 금지 - 팩트만 서술
"""

        try:
            result = self.llm.chat(
                messages=[
                    {"role": "system", "content": "당신은 금융 리서치 분석가입니다."},
                    {"role": "user", "content": analyze_prompt},
                ],
                max_tokens=1000,
                temperature=0.3,
            )

            if result.get("success"):
                import json
                import re

                content = result.get("content", "")
                # JSON 추출
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    what_is_missing = parsed.get("what_is_missing", [])

                    return {
                        "findings": parsed.get("findings", []),
                        "open_questions": parsed.get("open_questions", []),
                        "what_is_missing": what_is_missing,
                    }

        except Exception as e:
            logger.warning(f"Analyze 실행 실패: {e}")

        return {"findings": [], "open_questions": [], "what_is_missing": []}

    def _summarize_search_results(self, sources: List[Source]) -> str:
        """검색 결과 요약 생성

        Args:
            sources: 검색된 소스 리스트

        Returns:
            요약 문자열
        """
        if not sources:
            return "검색 결과 없음"

        # 소스 타입별 카운트
        type_counts = {}
        for src in sources:
            t = src.source_type
            type_counts[t] = type_counts.get(t, 0) + 1

        parts = [f"{len(sources)}개 소스 발견:"]
        for t, count in type_counts.items():
            parts.append(f"- {t}: {count}개")

        # 상위 3개 제목
        if sources:
            parts.append("상위 소스:")
            for src in sources[:3]:
                parts.append(f"  • {src.title[:50]}...")

        return "\n".join(parts)

    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """중복 소스 제거 (URL 기준)

        Args:
            sources: 원본 소스 리스트

        Returns:
            중복 제거된 리스트
        """
        seen_urls = set()
        unique = []

        for src in sources:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                unique.append(src)

        return unique

    def _create_trajectory_step(
        self,
        step_num: int,
        pre_sources: List[Source],
        post_sources: List[Source],
        all_search_queries: List[str],
        step: ReasoningStep,
        new_source_ids: List[str],
        rationale: str,
        consecutive_search_count: int = 0,
        pre_todo_list: Optional[List[ResearchTodoItem]] = None,
        post_todo_list: Optional[List[ResearchTodoItem]] = None,
    ) -> TrajectoryStep:
        """Trajectory 스텝 생성 (Policy 학습용)

        Args:
            step_num: 스텝 번호
            pre_sources: 액션 실행 전 소스 (pre_state용)
            post_sources: 액션 실행 후 소스 (post_state용)
            all_search_queries: 전체 검색 쿼리 기록
            step: 현재 추론 스텝
            new_source_ids: 이번 스텝에서 추가된 소스 ID
            rationale: 결정 근거
            consecutive_search_count: 연속 search 횟수
            pre_todo_list: 액션 실행 전 TODO 리스트
            post_todo_list: 액션 실행 후 TODO 리스트

        Returns:
            TrajectoryStep 객체
        """
        # 중복 제거된 소스 수 계산
        pre_unique = len(set(s.url for s in pre_sources))
        post_unique = len(set(s.url for s in post_sources))

        # TODO 관련 계산
        pre_todos = pre_todo_list or []
        post_todos = post_todo_list or []
        pre_completed_ids = [t.id for t in pre_todos if t.status == "done"]
        post_completed_ids = [t.id for t in post_todos if t.status == "done"]

        # 액션 실행 전 상태 (pre_state)
        pre_state = TrajectoryState(
            sources_count=len(pre_sources),
            unique_sources_count=pre_unique,
            search_queries_all=all_search_queries[:-1] if all_search_queries else [],
            search_queries_recent=(
                all_search_queries[-4:-1] if len(all_search_queries) > 1 else []
            ),
            consecutive_search_count=consecutive_search_count,
            open_questions=[],
            hypotheses=[],
            todo_list=pre_todos,
            completed_todo_ids=pre_completed_ids,
        )

        # 액션 실행 후 상태 (post_state)
        post_state = TrajectoryState(
            sources_count=len(post_sources),
            unique_sources_count=post_unique,
            search_queries_all=all_search_queries,
            search_queries_recent=all_search_queries[-3:] if all_search_queries else [],
            consecutive_search_count=(
                consecutive_search_count + 1 if step.action == ActionType.SEARCH else 0
            ),
            open_questions=[],
            hypotheses=[],
            todo_list=post_todos,
            completed_todo_ids=post_completed_ids,
        )

        # 관찰 결과
        top_snippets = []
        for sid, src in zip(
            new_source_ids,
            post_sources[-len(new_source_ids) :] if new_source_ids else [],
        ):
            top_snippets.append(
                {
                    "source_id": sid,
                    "snippet": src.snippet[:200] if src.snippet else "",
                }
            )

        observation = TrajectoryObservation(
            added_source_ids=new_source_ids,
            top_snippets=top_snippets[:3],
        )

        return TrajectoryStep(
            step=step_num,
            pre_state=pre_state,
            action=str(step.action),
            action_input=step.query,
            observation=observation,
            post_state=post_state,
            rationale=rationale,
            confidence=step.confidence,
        )

    def _create_claim_evidence_map(
        self,
        key_findings: List[str],
        sources: List[Source],
        conclusion: str,
        stock_name: Optional[str] = None,
    ) -> List[ClaimEvidence]:
        """Claim-Evidence 매핑 생성 (Writer 학습용)

        각 key_finding을 claim으로, 관련 소스를 evidence로 매핑
        vNext: evidence 없으면 해당 claim 자체를 제외 (fallback "상위 2개" 금지)
        vNext: 엔티티 mismatch evidence 필터링 (삼성전자 claim에 SK하이닉스 evidence 금지)

        Args:
            key_findings: 핵심 발견사항 리스트
            sources: 수집된 소스 리스트
            conclusion: 최종 결론
            stock_name: 대상 종목명 (엔티티 필터링용)

        Returns:
            ClaimEvidence 리스트
        """
        claim_evidence_list: List[ClaimEvidence] = []

        # 주요 기업 엔티티 목록 (claim-evidence mismatch 방지용)
        MAJOR_ENTITIES = [
            "삼성전자",
            "SK하이닉스",
            "LG에너지솔루션",
            "삼성SDI",
            "현대차",
            "기아",
            "네이버",
            "카카오",
            "삼성바이오로직스",
            "셀트리온",
            "포스코",
            "현대모비스",
            "LG화학",
            "SK이노베이션",
        ]

        def extract_entities(text: str) -> set:
            """텍스트에서 기업 엔티티 추출"""
            found = set()
            for entity in MAJOR_ENTITIES:
                if entity in text:
                    found.add(entity)
            return found

        def entities_compatible(
            claim_entities: set, evidence_entities: set, target_stock: Optional[str]
        ) -> bool:
            """엔티티 호환성 검사

            - 동일 엔티티가 있으면 호환
            - target_stock이 claim에 있고 evidence에 없으면 비호환
            - 경쟁사 언급(삼성↔SK하이닉스)은 비교 맥락으로 허용
            """
            if not claim_entities:
                return True  # claim에 특정 기업 없으면 모든 evidence 허용

            # target_stock이 claim에 있는 경우
            if target_stock and target_stock in claim_entities:
                # evidence에도 target_stock이 있거나, 빈 엔티티면 허용
                # evidence에 다른 기업만 언급되면 비호환 (엔티티 오염)
                return target_stock in evidence_entities or not evidence_entities

            # 일반적인 교집합 검사
            return bool(claim_entities & evidence_entities) or not evidence_entities

        # 각 key_finding에 대해 관련 소스 매핑
        for finding in key_findings:
            evidence = []
            finding_lower = finding.lower()

            # claim에서 엔티티 추출
            claim_entities = extract_entities(finding)

            # 간단한 키워드 매칭으로 관련 소스 찾기
            for src in sources:
                # 소스 텍스트 (제목 + snippet)
                src_text = f"{src.title} {src.snippet}"
                src_text_lower = src_text.lower()

                # 소스에서 엔티티 추출
                evidence_entities = extract_entities(src_text)

                # 엔티티 호환성 검사 (mismatch 필터링)
                if not entities_compatible(
                    claim_entities, evidence_entities, stock_name
                ):
                    logger.debug(
                        f"엔티티 mismatch 제외: claim={claim_entities}, evidence={evidence_entities}"
                    )
                    continue

                # 키워드 매칭 (finding의 핵심 단어가 소스에 있는지)
                finding_words = [w for w in finding_lower.split() if len(w) > 2]
                match_count = sum(1 for w in finding_words if w in src_text_lower)

                # 30% 이상 단어 매칭 시 관련 소스로 간주
                if finding_words and match_count / len(finding_words) >= 0.3:
                    evidence.append(
                        {
                            "source_id": src.source_id,  # stable hash-based ID
                            "support": src.snippet[:150] if src.snippet else src.title,
                        }
                    )

            # vNext: evidence 없으면 해당 claim 제외 (fallback 금지)
            if evidence:
                claim_evidence_list.append(
                    ClaimEvidence(
                        claim=finding,
                        evidence=evidence[:3],  # 최대 3개
                        risk=None,
                    )
                )
            else:
                logger.debug(f"Evidence 없는 claim 제외: {finding[:50]}...")

        # 최종 결론도 추가 (모든 소스를 근거로)
        if conclusion and sources:
            all_evidence = [
                {"source_id": src.source_id, "support": src.title}
                for src in sources[:5]
            ]
            claim_evidence_list.append(
                ClaimEvidence(
                    claim=(
                        conclusion[:200] + "..."
                        if len(conclusion) > 200
                        else conclusion
                    ),
                    evidence=all_evidence,
                    risk="종합 결론 - 개별 소스 해석 필요",
                )
            )

        return claim_evidence_list

    def _run_verification(
        self,
        result: ResearchResult,
        options: ResearchOptions,
    ) -> VerificationResult:
        """Adversary Agent 검증 실행

        vNext: fail-open 제거 - 검증 실패 시 verdict=None으로 표시

        Args:
            result: 검증할 ResearchResult
            options: 리서치 옵션

        Returns:
            VerificationResult (실패 시 verdict=None, error 필드에 원인 기록)
        """
        from .adversary import AdversaryAgent

        verification_start = time.time()

        try:
            adversary = AdversaryAgent(model=options.verification_model)
            verification = adversary.verify(result)

            verification.elapsed_ms = int((time.time() - verification_start) * 1000)

            logger.info(
                f"Adversary 검증 완료: verdict={verification.verdict}, "
                f"issues={len(verification.issues)}"
            )

            return verification

        except Exception as e:
            logger.error(f"Adversary 검증 실패: {e}")
            # vNext: fail-open 제거 - 검증 실패를 명시적으로 기록
            # verdict를 None처럼 처리할 수 없으므로 REVISE + 에러 메시지로 표시
            elapsed_ms = int((time.time() - verification_start) * 1000)
            return VerificationResult(
                verdict=VerificationVerdict.REVISE,  # 검증 실패 = 수정 필요
                issues=[],
                confidence=0.0,
                elapsed_ms=elapsed_ms,
                additional_sources=[{"error": str(e), "verification_failed": "true"}],
            )

    def research_request(self, request: ResearchRequest) -> ResearchResponse:
        """API 요청 처리

        Args:
            request: ResearchRequest 객체

        Returns:
            ResearchResponse 객체
        """
        try:
            result = self.research(
                query=request.query,
                options=request.options,
                stock_code=request.stock_code,
                stock_name=request.stock_name,
            )

            return ResearchResponse(
                success=True,
                result=result,
            )

        except Exception as e:
            logger.error(f"리서치 요청 처리 실패: {e}")
            return ResearchResponse(
                success=False,
                error=str(e),
            )

    @staticmethod
    def save_with_metadata(
        result: ResearchResult,
        output_path: Path,
        save_markdown: bool = False,
    ) -> Dict[str, Path]:
        """리서치 결과와 메타데이터를 별도 파일로 저장

        Args:
            result: ResearchResult 객체
            output_path: 기본 출력 경로 (예: result.json)
            save_markdown: 마크다운으로 저장 여부

        Returns:
            저장된 파일 경로 딕셔너리 {"result": Path, "metadata": Path}
        """
        import json

        output_path = Path(output_path)
        base_name = output_path.stem
        parent_dir = output_path.parent

        # 디렉토리 생성
        parent_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. 메인 결과 저장 (결론 + 소스)
        if save_markdown:
            result_path = parent_dir / f"{base_name}.md"
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(result.to_markdown())
        else:
            result_path = parent_dir / f"{base_name}.json"
            # 메타데이터 제외한 결과 (학습용 분리)
            result_data = {
                "query": result.query,
                "conclusion": result.conclusion,
                "key_findings": result.key_findings,
                "confidence": result.confidence,
                "reasoning_trace": [
                    {
                        "step": s.step_number,
                        "thought": s.thought,
                        "action": str(s.action),
                        "query": s.query,
                        "observation": s.observation,
                        "confidence": s.confidence,
                    }
                    for s in result.reasoning_trace
                ],
                "sources": [
                    {
                        "source_id": s.source_id,  # stable hash-based ID
                        "title": s.title,
                        "url": s.url,
                        "type": str(s.source_type),
                        "date": s.date,
                        "snippet": s.snippet[:200] if s.snippet else "",
                    }
                    for s in result.sources
                ],
            }
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

        saved_files["result"] = result_path

        # 2. 학습용 메타데이터 별도 저장
        metadata_path = parent_dir / f"{base_name}_metadata.json"
        metadata_data = {
            "query": result.query,
            "generated_at": result.metadata.started_at,
            "completed_at": result.metadata.completed_at,
            "elapsed_seconds": result.metadata.elapsed_seconds,
            "model_name": result.metadata.model_name,
            "iterations": result.metadata.iterations,
            "llm_calls": result.metadata.llm_calls,
            "sources_count": result.metadata.sources_count,
            "sources_by_type": result.metadata.sources_by_type,
            "search_queries": result.metadata.search_queries,
            "action_sequence": result.metadata.action_sequence,
            "avg_confidence": result.metadata.avg_confidence,
            "final_confidence": result.metadata.final_confidence,
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "input": tc.input_params,
                    "output": tc.output_summary,
                    "success": tc.success,
                    "elapsed_ms": tc.elapsed_ms,
                    "timestamp": tc.timestamp,
                }
                for tc in result.metadata.tool_calls
            ],
            "warnings": result.metadata.warnings,
            "error": result.metadata.error,
            # ============================================================
            # 학습용 데이터 (Policy + Writer)
            # vNext: pre_state → action → post_state 매핑
            # ============================================================
            "trajectory": [
                {
                    "step": ts.step,
                    "pre_state": {
                        "sources_count": ts.pre_state.sources_count,
                        "unique_sources_count": ts.pre_state.unique_sources_count,
                        "search_queries_recent": ts.pre_state.search_queries_recent,
                        "consecutive_search_count": ts.pre_state.consecutive_search_count,
                        "open_questions": ts.pre_state.open_questions,
                        "hypotheses": ts.pre_state.hypotheses,
                    },
                    "action": ts.action,
                    "action_input": ts.action_input,
                    "observation": {
                        "added_source_ids": ts.observation.added_source_ids,
                        "top_snippets": ts.observation.top_snippets,
                    },
                    "post_state": {
                        "sources_count": ts.post_state.sources_count,
                        "unique_sources_count": ts.post_state.unique_sources_count,
                        "search_queries_recent": ts.post_state.search_queries_recent,
                        "consecutive_search_count": ts.post_state.consecutive_search_count,
                        "open_questions": ts.post_state.open_questions,
                        "hypotheses": ts.post_state.hypotheses,
                    },
                    "rationale": ts.rationale,
                    "confidence": ts.confidence,
                }
                for ts in result.metadata.trajectory
            ],
            "claim_evidence_map": [
                {
                    "claim": ce.claim,
                    "evidence": ce.evidence,
                    "risk": ce.risk,
                }
                for ce in result.metadata.claim_evidence_map
            ],
            # ============================================================
            # Adversary Agent 검증 결과
            # ============================================================
            "verification": (
                {
                    "verdict": result.metadata.verification.verdict,
                    "issues": [
                        {
                            "type": issue.type,
                            "severity": issue.severity,
                            "claim_id": issue.claim_id,
                            "why": issue.why,
                            "citations": issue.citations,
                            "suggested_edit": issue.suggested_edit,
                        }
                        for issue in result.metadata.verification.issues
                    ],
                    "suggested_counter_queries": result.metadata.verification.suggested_counter_queries,
                    "confidence": result.metadata.verification.confidence,
                    "verified_at": result.metadata.verification.verified_at,
                    "perplexity_model": result.metadata.verification.perplexity_model,
                    "elapsed_ms": result.metadata.verification.elapsed_ms,
                    "additional_sources": result.metadata.verification.additional_sources,
                }
                if result.metadata.verification
                else None
            ),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_data, f, ensure_ascii=False, indent=2)

        saved_files["metadata"] = metadata_path

        logger.info(f"결과 저장: {result_path}")
        logger.info(f"메타데이터 저장: {metadata_path}")

        return saved_files

    @staticmethod
    def save_reasoning_trace(
        result: ResearchResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Reasoning trace를 JSON 로그로 저장

        리포트 생성 시 자동 호출됨. 학습 데이터 및 디버깅용.

        Args:
            result: ResearchResult 객체
            output_dir: 저장 디렉토리 (기본: output/reasoning_traces/)

        Returns:
            저장된 파일 경로
        """
        # 기본 디렉토리 설정
        if output_dir is None:
            output_dir = PROJECT_ROOT / "output" / "reasoning_traces"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 파일명 생성: reasoning_trace_{timestamp}.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 쿼리에서 종목명/코드 추출 (파일명용)
        query_slug = result.query[:20].replace(" ", "_").replace("/", "_")
        filename = f"reasoning_trace_{query_slug}_{timestamp}.json"
        output_path = output_dir / filename

        # Reasoning trace 데이터 구조화
        trace_data = {
            "meta": {
                "query": result.query,
                "generated_at": datetime.now().isoformat(),
                "model": result.metadata.model_name,
                "total_iterations": result.metadata.iterations,
                "elapsed_seconds": result.metadata.elapsed_seconds,
                "final_confidence": result.confidence,
            },
            "reasoning_trace": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "action": str(step.action),
                    "query": step.query,
                    "observation": step.observation,
                    "sources_found": step.sources_found,
                    "confidence": step.confidence,
                }
                for step in result.reasoning_trace
            ],
            "trajectory": [
                {
                    "step": ts.step,
                    "pre_state": {
                        "sources_count": ts.pre_state.sources_count,
                        "consecutive_search_count": ts.pre_state.consecutive_search_count,
                        "search_queries_recent": ts.pre_state.search_queries_recent,
                    },
                    "action": ts.action,
                    "action_input": ts.action_input,
                    "observation": {
                        "added_source_ids": ts.observation.added_source_ids,
                        "top_snippets": ts.observation.top_snippets[:2],  # 상위 2개만
                    },
                    "post_state": {
                        "sources_count": ts.post_state.sources_count,
                        "consecutive_search_count": ts.post_state.consecutive_search_count,
                    },
                    "confidence": ts.confidence,
                    "rationale": ts.rationale[:100] if ts.rationale else "",
                }
                for ts in (result.trajectory or [])
            ],
            "sources_summary": {
                "total": len(result.sources),
                "by_type": result.metadata.sources_by_type,
                "search_queries": result.metadata.search_queries,
            },
            "conclusion_preview": result.conclusion[:500] if result.conclusion else "",
            "key_findings": result.key_findings,
            "data_gaps": result.data_gaps,
        }

        # JSON 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Reasoning trace 저장: {output_path}")
        return output_path


# =============================================================================
# CLI 인터페이스
# =============================================================================


def main():
    """CLI 진입점"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="VELA CoT Research Agent - Chain-of-Thought 리서치"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="리서치 쿼리 (예: 'SK하이닉스 HBM 시장 전망')",
    )
    parser.add_argument(
        "--max-iterations",
        "-i",
        type=int,
        default=5,
        help="최대 추론 반복 횟수 (기본: 5)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="runpod",
        choices=["runpod", "mlx", "vllm"],
        help="LLM 백엔드 (기본: runpod)",
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="콘텐츠 본문 추출 비활성화",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="결과 저장 파일 경로 (JSON)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="마크다운 형식으로 출력",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 로깅 활성화",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Adversary Agent 검증 활성화 (Perplexity API 필요)",
    )
    parser.add_argument(
        "--verify-model",
        type=str,
        default="sonar",
        help="검증에 사용할 Perplexity 모델 (기본: sonar)",
    )

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 옵션 설정
    options = ResearchOptions(
        max_iterations=args.max_iterations,
        extract_content=not args.no_content,
        enable_verification=args.verify,
        verification_model=args.verify_model,
    )

    # 에이전트 초기화
    agent = ResearchAgent(
        llm_backend=args.backend,
        extract_content=not args.no_content,
    )

    # 리서치 실행
    print(f"\n🔍 리서치 시작: {args.query}\n")
    result = agent.research(
        query=args.query,
        options=options,
    )

    # 결과 출력
    if args.markdown:
        print(result.to_markdown())
    else:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    # 파일 저장 (결과 + 메타데이터 별도)
    if args.output:
        output_path = Path(args.output)
        saved = ResearchAgent.save_with_metadata(
            result=result,
            output_path=output_path,
            save_markdown=args.markdown,
        )
        print(f"\n✅ 결과 저장: {saved['result']}")
        print(f"✅ 메타데이터 저장: {saved['metadata']} (학습용)")

    # 요약 출력
    print("\n📊 리서치 완료")
    print(f"   - 신뢰도: {result.confidence:.0%}")
    print(f"   - 소스 수: {len(result.sources)}개")
    print(f"   - 반복 횟수: {result.metadata.iterations}회")
    print(f"   - 도구 호출: {len(result.metadata.tool_calls)}회")
    print(f"   - 검색 쿼리: {len(result.metadata.search_queries)}개")
    print(f"   - 소요 시간: {result.metadata.elapsed_seconds:.1f}초")


if __name__ == "__main__":
    main()
