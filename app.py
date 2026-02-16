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


def run_research(query: str, max_iterations: int) -> tuple[str, str, str]:
    """리서치 실행 후 (markdown, reasoning, raw_json) 튜플 반환."""
    if not query or not query.strip():
        return "쿼리를 입력해주세요.", "", ""

    try:
        from vela import ResearchAgent
        from vela.schemas import ResearchOptions

        agent = ResearchAgent(llm_backend=BACKEND)
        options = ResearchOptions(
            max_iterations=int(max_iterations),
            extract_content=True,
        )

        result = agent.research(query=query.strip(), options=options)

        # 1) 마크다운 리포트
        markdown_report = result.to_markdown()

        # 2) 추론 과정
        reasoning_lines = []
        for step in result.reasoning_trace:
            reasoning_lines.append(f"### Step {step.step_number}")
            reasoning_lines.append(f"**Thought**: {step.thought}")
            reasoning_lines.append(f"**Action**: {step.action}")
            if step.query:
                reasoning_lines.append(f"**Query**: `{step.query}`")
            reasoning_lines.append(f"**Observation**: {step.observation}")
            reasoning_lines.append(f"**Confidence**: {step.confidence:.0%}")
            reasoning_lines.append("")
        reasoning_md = "\n".join(reasoning_lines) if reasoning_lines else "추론 과정 없음"

        # 3) Raw JSON
        raw_json = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

        return markdown_report, reasoning_md, raw_json

    except Exception as e:
        logger.error(f"리서치 실패: {e}")
        error_md = (
            f"## 오류 발생\n\n"
            f"```\n{type(e).__name__}: {e}\n```\n\n"
            f"<details><summary>Traceback</summary>\n\n"
            f"```\n{traceback.format_exc()}\n```\n\n"
            f"</details>"
        )
        return error_md, "", ""


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
