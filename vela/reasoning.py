"""CoT 추론 엔진 - TODO 기반 Chain-of-Thought 반복 추론

TODO 리스트 생성 → 하나씩 검색/분석 → 모든 critical 완료 → Conclude 패턴 구현
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schemas import ActionType, ReasoningStep, ResearchTodoItem, Source, TodoPriority
from .prompts import get_research_system_prompt, RESEARCH_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


# =============================================================================
# TODO 추론 결과 (ReasoningStep 확장)
# =============================================================================


@dataclass
class TodoReasoningResult:
    """TODO 기반 추론 결과

    ReasoningStep + TODO 상태 정보
    """

    step: ReasoningStep
    todo_list: List[ResearchTodoItem] = field(default_factory=list)
    todo_updates: List[Dict] = field(default_factory=list)  # Step 2+에서 변경사항
    current_todo_id: Optional[str] = None

    def get_critical_pending_count(self) -> int:
        """완료되지 않은 critical TODO 개수"""
        return sum(
            1
            for t in self.todo_list
            if t.priority == TodoPriority.CRITICAL and t.status != "done"
        )

    def get_completed_ids(self) -> List[str]:
        """완료된 TODO ID 목록"""
        return [t.id for t in self.todo_list if t.status == "done"]

    def all_critical_done(self) -> bool:
        """모든 critical TODO 완료 여부"""
        return self.get_critical_pending_count() == 0


# =============================================================================
# CoT 추론 엔진
# =============================================================================


class CoTReasoningEngine:
    """Chain-of-Thought 추론 엔진"""

    def __init__(
        self,
        llm_client,
        max_iterations: int = 5,
        min_confidence: float = 0.7,
    ):
        """
        Args:
            llm_client: LLM 클라이언트 (RunPodClient 호환)
            max_iterations: 최대 반복 횟수
            min_confidence: 조기 종료 신뢰도 임계값
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence

    def reason(
        self,
        query: str,
        context: Dict,
        step_number: int = 1,
    ) -> TodoReasoningResult:
        """단일 추론 스텝 실행 (TODO 기반)

        Args:
            query: 원본 리서치 쿼리
            context: 현재까지 수집된 컨텍스트
                - sources: List[Source]
                - previous_steps: List[ReasoningStep]
                - search_queries: List[str] (이미 사용한 쿼리)
                - todo_list: List[ResearchTodoItem] (Step 2+)
            step_number: 현재 스텝 번호

        Returns:
            TodoReasoningResult (step + todo_list + todo_updates)
        """
        start_time = time.time()

        # 컨텍스트 요약 생성 (TODO 상태 포함)
        context_summary = self._build_context_summary(query, context, step_number)

        # LLM 호출 (단일 스텝 형식)
        user_prompt = f"""리서치 주제: {query}

{context_summary}

다음 단계를 결정하세요. **3줄만 출력**:
**Thought**: """

        try:
            # 현재 날짜와 쿼리 주제가 포함된 시스템 프롬프트 사용
            query_subject = self._extract_query_subject(query)
            system_prompt = get_research_system_prompt(query_subject=query_subject)

            result = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1500,
                temperature=0.3,
            )

            if not result.get("success"):
                logger.error(f"LLM 호출 실패: {result.get('error')}")
                return self._fallback_todo_result(step_number, "LLM 호출 실패", context)

            content = result.get("content", "")
            parsed = self._parse_response(content)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # ReasoningStep 생성
            step = ReasoningStep(
                step_number=step_number,
                thought=parsed.get("thought", "파싱 실패"),
                action=ActionType(parsed.get("action", "search")),
                query=parsed.get("query"),
                observation="",  # 액션 실행 후 채워짐
                confidence=parsed.get("confidence", 0.5),
                elapsed_ms=elapsed_ms,
            )

            # TODO 파싱 (Step 1 vs Step 2+ 분기)
            todo_list = context.get("todo_list", [])
            todo_updates = []

            if step_number == 1:
                raw_todo_list = parsed.get("todo_list", [])
                todo_list = self._parse_todo_list(raw_todo_list, step_number)
                logger.info(f"Step 1: TODO {len(todo_list)}개 생성됨")
            else:
                raw_updates = parsed.get("todo_updates", [])
                todo_updates = raw_updates
                todo_list = self._apply_todo_updates(
                    todo_list, raw_updates, step_number
                )

            current_todo_id = parsed.get("current_todo_id")

            return TodoReasoningResult(
                step=step,
                todo_list=todo_list,
                todo_updates=todo_updates,
                current_todo_id=current_todo_id,
            )

        except Exception as e:
            logger.error(f"추론 실패: {e}")
            return self._fallback_todo_result(step_number, str(e), context)

    def should_continue(
        self,
        steps: List[ReasoningStep],
        current_confidence: float,
        todo_result: Optional[TodoReasoningResult] = None,
    ) -> bool:
        """추론 계속 여부 판단 (TODO 상태 고려)

        Args:
            steps: 지금까지의 추론 스텝
            current_confidence: 현재 신뢰도
            todo_result: TODO 추론 결과 (있으면 critical TODO 체크)

        Returns:
            True면 계속, False면 종료
        """
        # 최대 반복 횟수 체크
        if len(steps) >= self.max_iterations:
            logger.info(f"최대 반복 횟수 도달: {self.max_iterations}")
            return False

        # 마지막 액션이 conclude면 종료
        if steps and steps[-1].action == ActionType.CONCLUDE:
            if todo_result and not todo_result.all_critical_done():
                pending = todo_result.get_critical_pending_count()
                logger.warning(f"Conclude 요청했으나 critical TODO {pending}개 미완료")
            logger.info("Conclude 액션 - 종료")
            return False

        # TODO 기반 종료 조건: 모든 critical 완료 + 높은 신뢰도
        if todo_result and todo_result.all_critical_done():
            if current_confidence >= 0.85:
                logger.info(
                    f"모든 critical TODO 완료 + 높은 신뢰도 ({current_confidence:.0%}) - 종료 권장"
                )
                return False

        # 기존: 신뢰도만 체크 (TODO 없을 때)
        if current_confidence >= 0.9:
            logger.info(f"높은 신뢰도 ({current_confidence:.0%}) - 종료 권장")
            return False

        return True

    def synthesize(
        self,
        query: str,
        steps: List[ReasoningStep],
        sources: List[Source],
        what_is_missing: Optional[List[str]] = None,
        confirmed_data: Optional[Dict] = None,
    ) -> Dict:
        """최종 결론 합성

        Args:
            query: 원본 쿼리
            steps: 추론 과정
            sources: 수집된 소스
            what_is_missing: 미제공 데이터 목록
            confirmed_data: 가드레일 - 시스템이 자동 추출한 확보된 데이터

        Returns:
            {"conclusion": str, "key_findings": List[str], "confidence": float, "data_gaps": List[str]}
        """
        # 소스 요약 생성
        sources_summary = self._build_sources_summary(sources)

        # 추론 과정 요약
        reasoning_summary = "\n".join(
            f"Step {s.step_number}: {s.thought[:100]}..." for s in steps
        )

        # 가드레일: 확보된 데이터 섹션 (시스템이 자동 추출, LLM 판단 X)
        confirmed_data_section = ""
        if confirmed_data and confirmed_data.get("provided_fields"):
            parts = ["## 확보된 데이터 (자동 추출)"]
            parts.append(
                "**반드시 아래 수치를 결론에 포함하세요. '미제공'으로 표기하면 안 됩니다.**\n"
            )

            if val := confirmed_data.get("valuation"):
                val_parts = []
                if v := val.get("current_price"):
                    val_parts.append(f"현재가: {v}")
                if v := val.get("12m_fwd_per"):
                    val_parts.append(f"12M FWD PER: {v}")
                elif v := val.get("per_ttm"):
                    val_parts.append(f"PER(TTM): {v}")
                if v := val.get("pbr"):
                    val_parts.append(f"PBR: {v}")
                if v := val.get("12m_fwd_eps"):
                    val_parts.append(f"12M FWD EPS: {v}")
                if v := val.get("roe"):
                    val_parts.append(f"ROE: {v}")
                if val_parts:
                    parts.append(f"- 밸류에이션: {', '.join(val_parts)}")

            if inv := confirmed_data.get("investor"):
                inv_parts = []
                if v := inv.get("foreign_net"):
                    inv_parts.append(f"외국인: {v}")
                if v := inv.get("institution_net"):
                    inv_parts.append(f"기관: {v}")
                if inv_parts:
                    parts.append(f"- 수급: {', '.join(inv_parts)}")

            if biz := confirmed_data.get("business"):
                parts.append(f"- 사업: {biz[:100]}...")

            confirmed_data_section = "\n".join(parts) + "\n"

        # 미제공 데이터 섹션 (확보된 필드 제외)
        missing_data_section = ""
        if what_is_missing:
            provided = (
                set(confirmed_data.get("provided_fields", []))
                if confirmed_data
                else set()
            )

            field_mapping = {
                "12m_fwd_per": ["per", "fwd per", "12m per"],
                "per_ttm": ["per", "ttm per"],
                "pbr": ["pbr"],
                "12m_fwd_eps": ["eps", "fwd eps", "12m eps", "컨센서스"],
                "roe": ["roe"],
            }

            filtered_missing = []
            for item in what_is_missing:
                item_lower = item.lower()
                is_provided = False
                for field_key, keywords in field_mapping.items():
                    if field_key in provided:
                        for kw in keywords:
                            if kw in item_lower:
                                is_provided = True
                                break
                    if is_provided:
                        break
                if not is_provided:
                    filtered_missing.append(item)

            if filtered_missing:
                missing_data_section = f"""
## 미제공 데이터 (반드시 결론에 명시)
{chr(10).join(f'- {item}' for item in filtered_missing)}

위 항목들은 소스에서 확인되지 않았습니다. 결론에 "미확인" 또는 "미제공"으로 표기하세요.
"""

        synthesis_prompt = f"""{query} 분석 리포트를 작성해주세요.

## 수집된 데이터
{sources_summary}

{confirmed_data_section}

리포트 마지막에 반드시 다음 형식으로 신뢰도를 기재하세요:
## Confidence: N%

## Key Findings
- """

        try:
            result = self.llm.chat(
                messages=[
                    {"role": "system", "content": RESEARCH_SYNTHESIS_PROMPT},
                    {"role": "user", "content": synthesis_prompt},
                ],
                max_tokens=2048,
                temperature=0.5,
                stop=["\nH:", "\nQ:", "\nUser:", "\nHuman:"],
            )

            if result.get("success"):
                content = result.get("content", "")
                parsed = self._parse_synthesis_response(content)

                conclusion = parsed.get("conclusion", "결론 생성 실패")

                # 후처리: 확보된 데이터가 있으면 "### 미제공 데이터" 섹션 제거
                if confirmed_data and confirmed_data.get("provided_fields"):
                    conclusion = re.sub(
                        r"### 미제공 데이터.*?(?=###|\Z)",
                        "",
                        conclusion,
                        flags=re.DOTALL,
                    ).strip()

                # 후처리: 섹션 중복 제거 + boilerplate 제거
                conclusion = self._dedup_sections(conclusion)
                conclusion = self._remove_boilerplate(conclusion)
                conclusion = self._truncate_after_conclusion(conclusion)

                # 후처리: 인라인 잡음 제거
                conclusion = re.sub(
                    r"(?:^|\n)\s*#?Tag[s]?:.*?(?:\n|$)", "\n", conclusion
                )
                conclusion = re.sub(
                    r"(?:^|\n)\s*저작권자?:.*?(?:\n|$)", "\n", conclusion
                )
                conclusion = re.sub(r"\n{3,}", "\n\n", conclusion).strip()

                return {
                    "conclusion": conclusion,
                    "key_findings": parsed.get("key_findings", []),
                    "confidence": parsed.get("confidence", 0.5),
                    "data_gaps": parsed.get("data_gaps", what_is_missing or []),
                }

        except Exception as e:
            logger.error(f"결론 합성 실패: {e}")

        return {
            "conclusion": "결론 생성에 실패했습니다.",
            "key_findings": [],
            "confidence": 0.3,
            "data_gaps": what_is_missing or [],
        }

    @staticmethod
    def _dedup_sections(text: str) -> str:
        """마크다운 헤더 기준으로 섹션 중복 제거 (첫 번째만 유지)"""
        # 결론 유사 헤더 정규화 (결론, 최종 결론, 리포트 결론, conclusio 등)
        _CONCLUSION_VARIANTS = re.compile(
            r"^(최종\s*)?결론$|^리포트\s*결론$|^투자\s*결론$|^conclusio[n]?$",
            re.IGNORECASE,
        )

        lines = text.split("\n")
        seen_headers = set()
        result_lines = []
        skip_until_next_header = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                header_key = re.sub(r"\s+", " ", stripped.lstrip("#").strip()).lower()
                if header_key:
                    # 결론 유사 헤더는 모두 "결론"으로 정규화
                    if _CONCLUSION_VARIANTS.match(header_key):
                        header_key = "결론"
                    if header_key in seen_headers:
                        skip_until_next_header = True
                        continue
                    seen_headers.add(header_key)
                skip_until_next_header = False
            elif skip_until_next_header:
                continue
            result_lines.append(line)

        return "\n".join(result_lines)

    @staticmethod
    def _truncate_after_conclusion(text: str) -> str:
        """결론/Conclusion 섹션 이후의 잡음 제거

        7B 모델은 결론 이후 참고 문서, 테이블, 증권 리포트 텍스트 등
        무의미한 콘텐츠를 계속 생성함. 결론 본문까지만 유지.
        """
        _CONCLUSION_PAT = re.compile(
            r"^#{1,3}\s*(?:(?:최종\s*)?결론|투자\s*결론|투자\s*의견|"
            r"conclusio[n]?(?:s?\s*(?:and|&)\s*key\s*takeaways?)?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        matches = list(_CONCLUSION_PAT.finditer(text))
        if not matches:
            return text

        last_conclusion = matches[-1]
        after = text[last_conclusion.end():]

        # 1. 다음 헤더에서 자르기
        next_header = re.search(r"\n#{1,3}\s+", after)
        if next_header:
            return text[:last_conclusion.end() + next_header.start()].strip()

        # 2. --- 구분선에서 자르기
        separator = re.search(r"\n---", after)
        if separator:
            return text[:last_conclusion.end() + separator.start()].strip()

        # 3. 결론 본문이 500자 이상이면 첫 번째 빈 줄에서 자르기
        if len(after) > 500:
            blank_line = re.search(r"\n\n", after[200:])
            if blank_line:
                return text[:last_conclusion.end() + 200 + blank_line.start()].strip()

        return text

    @staticmethod
    def _remove_boilerplate(text: str) -> str:
        """7B 모델이 생성하는 boilerplate 섹션 제거"""
        _JUNK_HEADERS = re.compile(
            r"^(?:legal(?:\s*(?:&|and)\s*compliance|\s*disclaimer)?|disclaimer|"
            r"contact(?:\s*information)?|tag[s]?|action\s*plan|copyright|"
            r"저작권|면책|연락처|태그|핵심\s*키워드|키워드)$",
            re.IGNORECASE,
        )

        lines = text.split("\n")
        result_lines = []
        skip_until_next_header = False
        current_section_lines = []
        current_header = None

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                # 이전 섹션 flush
                if current_header is not None:
                    header_text = re.sub(r"\s+", " ", current_header.lstrip("#").strip())
                    body = "\n".join(current_section_lines).strip()
                    # 2글자 이하 헤더 or junk 헤더 or 본문 30자 미만 → 스킵
                    if (len(header_text) <= 2
                            or _JUNK_HEADERS.match(header_text.lower())
                            or (len(body) < 30 and not body)):
                        pass  # 스킵
                    else:
                        result_lines.append(current_header)
                        result_lines.extend(current_section_lines)

                current_header = line
                current_section_lines = []
                skip_until_next_header = False
            else:
                current_section_lines.append(line)

        # 마지막 섹션 flush
        if current_header is not None:
            header_text = re.sub(r"\s+", " ", current_header.lstrip("#").strip())
            body = "\n".join(current_section_lines).strip()
            if not (len(header_text) <= 2
                    or _JUNK_HEADERS.match(header_text.lower())
                    or (len(body) < 30 and not body)):
                result_lines.append(current_header)
                result_lines.extend(current_section_lines)

        return "\n".join(result_lines)

    def _extract_query_subject(self, query: str) -> str:
        """쿼리에서 주요 주제/종목 추출"""
        stock_patterns = [
            r"(삼성전자|SK하이닉스|현대차|LG에너지솔루션|삼성바이오로직스|NAVER|카카오|삼성SDI|현대모비스|POSCO홀딩스|KB금융|신한지주|삼성물산|하나금융지주|기아|삼성생명|셀트리온|LG화학|한국전력|SK이노베이션)",
            r"([가-힣]+전자|[가-힣]+반도체|[가-힣]+바이오|[가-힣]+제약|[가-힣]+금융|[가-힣]+증권|[가-힣]+은행)",
        ]

        for pattern in stock_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        words = query.split()
        if words:
            subject = " ".join(words[:3])
            return subject[:20]

        return query[:20]

    def _build_context_summary(
        self, query: str, context: Dict, step_number: int = 1
    ) -> str:
        """컨텍스트 요약 생성 (TODO 상태 포함)"""
        parts = []

        if step_number == 1:
            parts.append(
                "첫 번째 스텝입니다. TODO 리스트를 생성하고 첫 번째 작업을 시작하세요."
            )
        else:
            todo_list = context.get("todo_list", [])
            if todo_list:
                parts.append("## 현재 TODO 리스트")
                for todo in todo_list:
                    status_icon = {
                        "pending": "[ ]",
                        "in_progress": "[~]",
                        "done": "[x]",
                        "skipped": "[-]",
                    }.get(todo.status, "[ ]")
                    priority_tag = (
                        f"[{todo.priority}]" if hasattr(todo, "priority") else ""
                    )
                    summary = (
                        f" - {todo.result_summary[:50]}..."
                        if todo.result_summary
                        else ""
                    )
                    parts.append(
                        f"{status_icon} {todo.id}: {priority_tag} {todo.task}{summary}"
                    )

                if isinstance(todo_list[0], ResearchTodoItem):
                    critical_pending = sum(
                        1
                        for t in todo_list
                        if t.priority == TodoPriority.CRITICAL and t.status != "done"
                    )
                    done_count = sum(1 for t in todo_list if t.status == "done")
                    parts.append(
                        f"\n진행 상황: {done_count}/{len(todo_list)} 완료, "
                        f"critical 미완료: {critical_pending}개"
                    )

        search_queries = context.get("search_queries", [])
        if search_queries:
            parts.append(f"\n이미 검색한 쿼리: {', '.join(search_queries[-5:])}")

        sources = context.get("sources", [])
        if sources:
            news_count = sum(1 for s in sources if s.source_type == "news")
            report_count = sum(1 for s in sources if s.source_type == "report")
            parts.append(f"수집된 소스: 뉴스 {news_count}개, 리포트 {report_count}개")

            recent_titles = [s.title[:50] for s in sources[-3:]]
            parts.append(f"최근 소스: {', '.join(recent_titles)}")

        previous_steps = context.get("previous_steps", [])
        if previous_steps:
            last_step = previous_steps[-1]
            parts.append(
                f"\n이전 스텝: {last_step.action} (신뢰도: {last_step.confidence:.0%})"
            )
            if last_step.observation:
                parts.append(f"관찰: {last_step.observation[:100]}...")

        return "\n".join(parts) if parts else "첫 번째 검색 단계"

    def _build_sources_summary(self, sources: List[Source]) -> str:
        """소스 목록 요약 (PRICE/INVESTOR는 confirmed_data로 처리되므로 제외)"""
        if not sources:
            return "수집된 소스 없음"

        # PRICE/INVESTOR 소스는 confirmed_data_section에서 구조화하여 주입
        _SKIP_TYPES = {"price", "investor"}
        news_sources = [s for s in sources if s.source_type not in _SKIP_TYPES]

        lines = []
        for i, src in enumerate(news_sources[:10], 1):
            line = f"{i}. [{src.source_type}] {src.title[:60]}"
            if src.date:
                line += f" ({src.date})"
            if src.snippet:
                line += f"\n   {src.snippet[:100]}..."
            lines.append(line)

        return "\n".join(lines)

    def _parse_response(self, content: str) -> Dict:
        """LLM 응답 파싱 (Markdown 형식)

        출력 형식:
        **Thought**: [분석 내용]
        **Action**: search
        **Query**: 검색어
        **Confidence**: X%
        """
        result = {}

        # 1. **Thought**: 내용
        thought_match = re.search(
            r"\*\*Thought\*\*:\s*(.+?)(?=\n\*\*|\Z)", content, re.DOTALL
        )
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 2. **Action**: search 또는 conclude
        action_match = re.search(r"\*\*Action\*\*:\s*(\w+)", content)
        if action_match:
            action_str = action_match.group(1).lower()
            if action_str in ("search", "analyze", "conclude"):
                result["action"] = action_str
            else:
                result["action"] = "search"

        # 3. **Query**: 검색어 (별도 줄)
        query_match = re.search(r"\*\*Query\*\*:\s*(.+?)(?=\n\*\*|\n\n|\Z)", content)
        if query_match:
            query = query_match.group(1).strip()
            if query.lower() not in ("없음", "null", "none", "-", ""):
                result["query"] = query

        # 3b. Fallback: (query: ...) 형식도 시도
        if not result.get("query"):
            inline_query = re.search(r"\(query:\s*(.+?)\)", content, re.IGNORECASE)
            if inline_query:
                result["query"] = inline_query.group(1).strip()

        # 4. **Confidence**: 85%
        conf_match = re.search(r"\*\*Confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", content)
        if conf_match:
            conf_val = float(conf_match.group(1))
            result["confidence"] = conf_val / 100 if conf_val > 1 else conf_val

        # 프롬프트가 "**Thought**: "로 끝나므로 모델 출력에 **Thought**: 없을 수 있음
        # → **Action**: 앞 텍스트를 thought로 추출
        if not result.get("thought") and result.get("action"):
            pre_action = re.search(
                r"^(.*?)(?=\*\*Action\*\*:)", content, re.DOTALL
            )
            if pre_action and pre_action.group(1).strip():
                result["thought"] = pre_action.group(1).strip()[:500]
            else:
                result["thought"] = content[:200].strip() or "추론 중"

        # Markdown 파싱 성공
        if result.get("thought") and result.get("action"):
            logger.debug(f"Markdown 파싱 성공: {list(result.keys())}")
            if "confidence" not in result:
                result["confidence"] = 0.5
            return result

        # 2. JSON 파싱 fallback (이전 버전 호환)
        json_block = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # 중첩 괄호를 고려한 JSON 객체 추출
        start = content.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(content[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(content[start : i + 1])
                            logger.debug(f"JSON 파싱 성공: {list(parsed.keys())}")
                            return parsed
                        except json.JSONDecodeError:
                            break

        # 3. 단순 패턴 fallback
        logger.debug("구조화된 파싱 실패, 단순 패턴 시도")

        if "conclude" in content.lower() or "결론" in content:
            result["action"] = "conclude"
        elif "search" in content.lower() or "검색" in content:
            result["action"] = "search"
        else:
            result["action"] = "search"

        query_simple = re.search(
            r'query[:\s]+["\']?([^"\']+)["\']?', content, re.IGNORECASE
        )
        if query_simple:
            result["query"] = query_simple.group(1).strip()[:100]

        result["thought"] = content[:200] if content else "파싱 실패"
        result["confidence"] = 0.5

        logger.warning(f"Fallback 파싱: {content[:100]}...")
        return result

    def _parse_synthesis_response(self, content: str) -> Dict:
        """Synthesis 응답 파싱 (Markdown 형식)"""
        result = {}

        # 1. Key Findings 추출
        key_findings = []
        findings_match = re.search(
            r"##\s*Key\s*Findings?\s*\n([\s\S]*?)(?=\n---|\n##|\Z)",
            content,
            re.IGNORECASE,
        )
        if findings_match:
            findings_text = findings_match.group(1)
            findings = re.findall(r"[-*]\s+(.+?)(?=\n[-*]|\n\n|\Z)", findings_text)
            if not findings:
                findings = re.findall(
                    r"\d+\.\s+(.+?)(?=\n\d+\.|\n\n|\Z)", findings_text
                )
            key_findings = [f.strip() for f in findings if f.strip()]

        result["key_findings"] = key_findings[:5]

        # 2. Confidence 추출 (## Confidence: N% 또는 **Confidence: N%** 둘 다 지원)
        conf_match = re.search(
            r"(?:##?\s*|\*\*\s*)Confidence[:\s]*(\d+(?:\.\d+)?)\s*%",
            content,
            re.IGNORECASE,
        )
        if conf_match:
            conf_val = float(conf_match.group(1))
            result["confidence"] = conf_val / 100 if conf_val > 1 else conf_val
        else:
            result["confidence"] = 0.5

        # 3. Conclusion (전체 리포트 본문)
        conclusion_match = re.search(
            r"(##\s*.+?리[서포]트[\s\S]*?)(?=##\s*Confidence|\Z)",
            content,
            re.IGNORECASE,
        )
        if conclusion_match:
            result["conclusion"] = conclusion_match.group(1).strip()
        else:
            if findings_match:
                rest_start = findings_match.end()
                result["conclusion"] = content[rest_start:].strip()
            else:
                result["conclusion"] = content.strip()

        # 4. JSON 형식 fallback (이전 버전 호환)
        if not result.get("conclusion") or len(result["conclusion"]) < 50:
            json_match = re.search(r'\{[\s\S]*"conclusion"[\s\S]*\}', content)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if parsed.get("conclusion"):
                        result["conclusion"] = parsed["conclusion"]
                    if parsed.get("key_findings"):
                        result["key_findings"] = parsed["key_findings"]
                    if parsed.get("confidence"):
                        result["confidence"] = parsed["confidence"]
                except json.JSONDecodeError:
                    pass

        logger.debug(
            f"Synthesis 파싱: conclusion={len(result.get('conclusion', ''))}자, "
            f"findings={len(result.get('key_findings', []))}개"
        )

        return result

    def _fallback_step(self, step_number: int, error_msg: str) -> ReasoningStep:
        """에러 시 폴백 스텝 생성"""
        return ReasoningStep(
            step_number=step_number,
            thought=f"에러 발생: {error_msg}",
            action=ActionType.CONCLUDE,
            query=None,
            observation="에러로 인한 조기 종료",
            confidence=0.3,
        )

    # =========================================================================
    # TODO 파싱 헬퍼 메서드
    # =========================================================================

    def _parse_todo_list(
        self, raw_todo_list: List[Dict], step_number: int
    ) -> List[ResearchTodoItem]:
        """Step 1 응답에서 TODO 리스트 파싱"""
        parsed_todos = []

        for raw in raw_todo_list:
            try:
                todo_id = raw.get("id", f"t{len(parsed_todos) + 1}")
                task = raw.get("task", "작업 설명 없음")
                priority_str = raw.get("priority", "medium").lower()
                status = raw.get("status", "pending")

                priority_map = {
                    "critical": TodoPriority.CRITICAL,
                    "high": TodoPriority.HIGH,
                    "medium": TodoPriority.MEDIUM,
                    "low": TodoPriority.LOW,
                }
                priority = priority_map.get(priority_str, TodoPriority.MEDIUM)

                todo_item = ResearchTodoItem(
                    id=todo_id,
                    task=task,
                    priority=priority,
                    status=status,
                    created_at_step=step_number,
                    search_query=raw.get("search_query"),
                    result_summary=raw.get("result_summary"),
                )
                parsed_todos.append(todo_item)

            except Exception as e:
                logger.warning(f"TODO 파싱 실패: {raw} - {e}")
                continue

        if not parsed_todos:
            logger.warning("TODO 리스트가 비어있어 기본 TODO 생성")
            parsed_todos.append(
                ResearchTodoItem(
                    id="t1",
                    task="기본 정보 수집",
                    priority=TodoPriority.CRITICAL,
                    status="in_progress",
                    created_at_step=step_number,
                )
            )

        return parsed_todos

    def _apply_todo_updates(
        self,
        todo_list: List[ResearchTodoItem],
        raw_updates: List[Dict],
        step_number: int,
    ) -> List[ResearchTodoItem]:
        """Step 2+ 응답에서 TODO 업데이트 적용"""
        if not raw_updates:
            return todo_list

        todo_map = {t.id: t for t in todo_list}

        for update in raw_updates:
            try:
                todo_id = update.get("id")
                if not todo_id or todo_id not in todo_map:
                    logger.warning(f"존재하지 않는 TODO ID: {todo_id}")
                    continue

                todo = todo_map[todo_id]

                new_status = update.get("status")
                if new_status and new_status in [
                    "pending",
                    "in_progress",
                    "done",
                    "skipped",
                ]:
                    old_status = todo.status
                    todo.status = new_status

                    if new_status == "done" and old_status != "done":
                        todo.completed_at_step = step_number

                if update.get("result_summary"):
                    todo.result_summary = update["result_summary"]

                logger.debug(f"TODO {todo_id} 업데이트: {update}")

            except Exception as e:
                logger.warning(f"TODO 업데이트 실패: {update} - {e}")
                continue

        return list(todo_map.values())

    def _fallback_todo_result(
        self, step_number: int, error_msg: str, context: Dict
    ) -> TodoReasoningResult:
        """에러 시 폴백 TODO 결과 생성"""
        fallback_step = self._fallback_step(step_number, error_msg)
        todo_list = context.get("todo_list", [])

        return TodoReasoningResult(
            step=fallback_step,
            todo_list=todo_list,
            todo_updates=[],
            current_todo_id=None,
        )
