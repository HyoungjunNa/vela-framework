"""VELA Research Agent - Gradio Web Demo

HuggingFace Spaces 배포용 Gradio 데모.
RunPod Serverless 백엔드로 VELA 7B 모델을 실행합니다.

HuggingFace Spaces 배포 시:
  1. Spaces 설정에서 SDK를 "gradio"로 선택
  2. Secrets에 환경변수 추가:
     - RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
     - NAVER_CLIENT_ID_0, NAVER_CLIENT_SECRET_0
  3. README.md 상단에 HF Spaces 메타데이터 추가 (sdk: gradio)
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


def check_api_keys() -> list[str]:
    """필수 API 키 설정 확인. 누락된 키 이름 리스트 반환."""
    missing = []
    if not os.environ.get("RUNPOD_API_KEY"):
        missing.append("RUNPOD_API_KEY")
    if not os.environ.get("RUNPOD_ENDPOINT_ID"):
        missing.append("RUNPOD_ENDPOINT_ID")
    return missing


def run_research(query: str, max_iterations: int) -> tuple[str, str, str]:
    """리서치 실행 후 (markdown, reasoning, raw_json) 튜플 반환."""
    if not query or not query.strip():
        return "쿼리를 입력해주세요.", "", ""

    # API 키 확인
    missing = check_api_keys()
    if missing:
        msg = (
            "## API 키 미설정\n\n"
            "다음 환경변수를 설정해주세요:\n\n"
            + "\n".join(f"- `{k}`" for k in missing)
            + "\n\nHuggingFace Spaces: Settings > Secrets에서 추가"
        )
        return msg, "", ""

    try:
        from vela import ResearchAgent
        from vela.schemas import ResearchOptions

        agent = ResearchAgent(llm_backend="runpod")
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

with gr.Blocks(title="VELA Research Agent") as demo:
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
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
