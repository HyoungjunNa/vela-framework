"""VELA Research Agent - Gradio Web Demo

HuggingFace Spaces 배포용 Gradio 데모.
ZeroGPU 백엔드로 VELA 7B 모델을 실행합니다 (HF Pro 필요).

HuggingFace Spaces 배포 시:
  1. Spaces 설정에서 SDK를 "gradio", Hardware를 "ZeroGPU"로 선택
  2. (선택) Secrets에 검색 API 키 추가:
     - NAVER_CLIENT_ID_1, NAVER_CLIENT_SECRET_1
  3. GPU는 @spaces.GPU 데코레이터로 자동 할당
"""

import json
import logging
import os
import time
import traceback

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_backend() -> str:
    """환경에 따른 LLM 백엔드 자동 선택"""
    if os.environ.get("VELA_LLM_BACKEND"):
        return os.environ["VELA_LLM_BACKEND"]
    if os.environ.get("SPACE_ID"):
        return "zerogpu"
    if os.environ.get("RUNPOD_API_KEY"):
        return "runpod"
    return "zerogpu"


BACKEND = get_backend()
logger.info(f"LLM 백엔드: {BACKEND}")

# ZeroGPU: 전체 research를 단일 @spaces.GPU(duration=300)으로 래핑
# _generate()마다 @spaces.GPU를 붙이면 동일 요청 내 두 번째 GPU 할당 실패
_has_spaces = False
if BACKEND == "zerogpu":
    import vela.tools.zerogpu_client  # noqa: F401 — 모델 사전 로드
    try:
        import spaces
        _has_spaces = True
    except ImportError:
        pass

if _has_spaces:
    @spaces.GPU(duration=300)
    def _run_research_gpu(agent, query, options, step_callback):
        """GPU 컨텍스트 내에서 전체 research 실행 (단일 GPU 할당)"""
        return agent.research(query=query, options=options, step_callback=step_callback)
else:
    def _run_research_gpu(agent, query, options, step_callback):
        return agent.research(query=query, options=options, step_callback=step_callback)


def run_research(query: str, max_iterations: int):
    """리서치 실행 — 스트리밍 제너레이터.

    ZeroGPU: 전체 research를 단일 @spaces.GPU(duration=300) 컨텍스트로 실행.
    동일 Gradio 요청 내 다중 @spaces.GPU 호출 시 두 번째부터 GPU 할당 실패하므로
    _run_research_gpu()에서 한 번만 GPU를 할당하고 모든 LLM 추론을 수행.
    """
    if not query or not query.strip():
        yield "쿼리를 입력해주세요.", "", ""
        return

    try:
        from vela import ResearchAgent
        from vela.schemas import ResearchOptions

        # 첫 번째 yield: 진행 상황 초기화 (UI 즉시 반응)
        progress_lines = [f"## 리서치 진행 중: {query.strip()}\n"]
        yield "\n".join(progress_lines), "", ""

        def on_step(info):
            phase = info.get("phase")
            step = info.get("step", "")
            if phase == "reasoning":
                progress_lines.append(f"### Step {step}")
                progress_lines.append("추론 중...")
            elif phase == "searching":
                q = info.get("query", "")
                progress_lines.append(f"검색: `{q}`")
            elif phase == "search_done":
                n = info.get("sources_found", 0)
                progress_lines.append(f"**{n}개** 소스 발견\n")
            elif phase == "synthesizing":
                n = info.get("sources_count", 0)
                progress_lines.append(f"\n### 최종 리포트 생성 중... ({n}개 소스 종합)")

        agent = ResearchAgent(llm_backend=BACKEND)
        options = ResearchOptions(
            max_iterations=int(max_iterations),
            extract_content=True,
        )
        # 단일 GPU 컨텍스트에서 전체 research 실행
        result = _run_research_gpu(agent, query.strip(), options, on_step)

        if not result:
            yield "리서치 결과가 없습니다.", "", ""
            return

        # 1) 마크다운 리포트
        markdown_report = result.to_markdown()

        # 2) 추론 과정
        reasoning_lines = []
        for s in result.reasoning_trace:
            reasoning_lines.append(f"### Step {s.step_number}")
            reasoning_lines.append(f"**Thought**: {s.thought}")
            reasoning_lines.append(f"**Action**: {s.action}")
            if s.query:
                reasoning_lines.append(f"**Query**: `{s.query}`")
            reasoning_lines.append(f"**Observation**: {s.observation}")
            reasoning_lines.append(f"**Confidence**: {s.confidence:.0%}")
            reasoning_lines.append("")
        reasoning_md = "\n".join(reasoning_lines) if reasoning_lines else "추론 과정 없음"

        # 3) Raw JSON
        raw_json = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

        yield markdown_report, reasoning_md, raw_json

    except Exception as e:
        logger.error(f"리서치 실패: {e}")
        error_md = (
            f"## 오류 발생\n\n"
            f"```\n{type(e).__name__}: {e}\n```\n\n"
            f"<details><summary>Traceback</summary>\n\n"
            f"```\n{traceback.format_exc()}\n```\n\n"
            f"</details>"
        )
        yield error_md, "", ""


# ============================================================================
# Gradio UI
# ============================================================================

EXAMPLES = [
    ["SK하이닉스 HBM 시장 전망", 3],
    ["삼성전자 파운드리 경쟁력 분석", 3],
    ["네이버 AI 사업 전략", 3],
    ["현대차 전기차 시장 점유율", 3],
]

with gr.Blocks(title="VELA Research Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# VELA Research Agent Demo\n"
        "*Korean Financial Research with 7B LLM*\n\n"
        "VELA는 한국 주식시장 전문 리서치 에이전트입니다. "
        "Chain-of-Thought 추론으로 웹 검색, 분석, 결론 도출을 자동 수행합니다."
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="리서치 쿼리",
                placeholder="예: SK하이닉스 HBM 시장 전망",
                lines=1,
            )
        with gr.Column(scale=1):
            max_iter_slider = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="최대 반복",
            )

    run_btn = gr.Button("리서치 실행", variant="primary", size="lg")

    # 결과 영역
    report_output = gr.Markdown(label="리서치 결과")

    with gr.Accordion("추론 과정 (Reasoning Trace)", open=False):
        reasoning_output = gr.Markdown()

    with gr.Accordion("Raw JSON", open=False):
        json_output = gr.Code(language="json")

    # 예제
    gr.Examples(
        examples=EXAMPLES,
        inputs=[query_input, max_iter_slider],
        label="예제 쿼리",
    )

    # Limitations
    with gr.Accordion("Limitations", open=False):
        gr.Markdown(
            "### Known Limitations\n\n"
            "*이 데모는 공개 검색 API + 네이버 증권 데이터를 사용합니다.*\n\n"
            "| 항목 | 설명 | 상용 배포 |\n"
            "|------|------|----------|\n"
            "| **모델 크기** | 7B 파라미터 — 복잡한 다단계 추론은 대형 모델 대비 품질 저하 가능 | |\n"
            "| **언어** | 한국 금융 도메인 전용 — 영어/다국어 쿼리는 품질 저하 | |\n"
            "| **시세/밸류에이션** | 네이버 증권 실시간 연동 (PER/PBR/EPS/수급) | FnGuide 추가 가능 |\n"
            "| **검색 범위** | Naver + DuckDuckGo — 유료 DB 접근 불가 | 증권사 리포트 연동 |\n"
            "| **콘텐츠 추출** | 검색 단계당 상위 3개만 본문 추출 | 전문 추출 가능 |\n"
            "| **반복 생성** | 7B 모델 특성상 출력 반복 가능 — 후처리로 완화 | |\n"
            "| **신뢰도** | 자기 보고 방식 (calibrated 아님) | |\n\n"
            "### Production Enhancements\n\n"
            "상용 배포에서 VELA는 다음을 추가 연동할 수 있습니다:\n"
            "- **FnGuide API**: 실시간 컨센서스, 목표가, 애널리스트 평점 (50개+ 증권사)\n"
            "- **증권사 리포트**: 주요 증권사 리포트 전문 추출\n"
            "- **재무제표**: 3개년+ 대차대조표, 현금흐름표, 손익계산서\n\n"
            "엔터프라이즈 문의: hello@intrect.io\n\n"
            "---\n\n"
            "**VELA는 투자 조언 도구가 아닙니다.** "
            "정보 제공/교육 목적으로만 사용하세요. 투자 판단은 전문가와 상담하시기 바랍니다."
        )

    # 이벤트 바인딩
    run_btn.click(
        fn=run_research,
        inputs=[query_input, max_iter_slider],
        outputs=[report_output, reasoning_output, json_output],
    )
    query_input.submit(
        fn=run_research,
        inputs=[query_input, max_iter_slider],
        outputs=[report_output, reasoning_output, json_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
