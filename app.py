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
import queue
import threading
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

# ZeroGPU: @spaces.GPU 데코레이터를 시작 시 등록해야 함
if BACKEND == "zerogpu":
    import vela.tools.zerogpu_client  # noqa: F401 — registers @spaces.GPU


def run_research(query: str, max_iterations: int):
    """리서치 실행 — 스트리밍 제너레이터.

    각 추론 단계마다 yield하여 UI에 실시간 진행 상황 표시.
    """
    if not query or not query.strip():
        yield "쿼리를 입력해주세요.", "", ""
        return

    try:
        from vela import ResearchAgent
        from vela.schemas import ResearchOptions

        # 진행 상황 큐 (agent callback → generator)
        progress_q = queue.Queue()
        result_holder = [None, None]  # [result, error]

        def on_step(info):
            progress_q.put(info)

        def _run():
            try:
                agent = ResearchAgent(llm_backend=BACKEND)
                options = ResearchOptions(
                    max_iterations=int(max_iterations),
                    extract_content=True,
                )
                result_holder[0] = agent.research(
                    query=query.strip(), options=options, step_callback=on_step,
                )
            except Exception as e:
                result_holder[1] = e
            finally:
                progress_q.put({"phase": "done"})

        # 백그라운드 스레드에서 리서치 실행
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # 진행 상황 표시
        progress_lines = [f"## 리서치 진행 중: {query.strip()}\n"]
        start_time = time.time()

        while True:
            try:
                info = progress_q.get(timeout=0.5)
            except queue.Empty:
                # 타임아웃 — 경과 시간 업데이트만
                elapsed = int(time.time() - start_time)
                status = "\n".join(progress_lines) + f"\n\n*{elapsed}초 경과...*"
                yield status, "", ""
                continue

            phase = info.get("phase")
            step = info.get("step", "")

            if phase == "done":
                break
            elif phase == "reasoning":
                progress_lines.append(f"### Step {step}")
                progress_lines.append("🤔 추론 중...")
            elif phase == "searching":
                q = info.get("query", "")
                progress_lines.append(f"🔍 검색: `{q}`")
            elif phase == "search_done":
                n = info.get("sources_found", 0)
                progress_lines.append(f"✅ **{n}개** 소스 발견\n")
            elif phase == "analyzing":
                progress_lines.append("📊 소스 분석 중...")
            elif phase == "concluding":
                progress_lines.append("📝 결론 도출 중...")
            elif phase == "synthesizing":
                n = info.get("sources_count", 0)
                progress_lines.append(f"\n### 최종 리포트 생성 중... ({n}개 소스 종합)")

            yield "\n".join(progress_lines), "", ""

        thread.join(timeout=10)

        # 최종 결과
        result = result_holder[0]
        error = result_holder[1]

        if error:
            raise error

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
