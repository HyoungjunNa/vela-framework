"""ZeroGPU Client - HuggingFace Spaces GPU 추론

HF Spaces 환경: @spaces.GPU + transformers 모듈 레벨 로드 (float16)
로컬 환경: HF Inference API 원격 추론 (GPU 불필요)
Sampling: generation_config.json 위임 (top_k=40, top_p=0.95, rep_penalty=1.0)

RunPodClient 호환 인터페이스:
    client = ZeroGPUClient()
    result = client.chat([{"role": "user", "content": "질문"}])
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_ON_SPACES = bool(os.environ.get("SPACE_ID"))
MODEL_ID = os.environ.get("VELA_MODEL_ID", "intrect/VELA")

# =============================================================================
# spaces 패키지를 CUDA 초기화 전에 먼저 import (ZeroGPU 필수)
# =============================================================================
_has_spaces = False
if _ON_SPACES:
    try:
        import spaces

        _has_spaces = True
    except ImportError:
        pass

# =============================================================================
# Spaces: 모듈 레벨에서 모델 사전 로드 (spaces import 후)
# =============================================================================
_model = None
_tokenizer = None

if _ON_SPACES:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"[ZeroGPU] 모델 로딩 시작: {MODEL_ID}")
    _load_start = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"[ZeroGPU] 모델 로딩 완료: {time.time() - _load_start:.1f}초")


# =============================================================================
# 내부 생성 함수 (GPU 관리는 app.py의 단일 @spaces.GPU 컨텍스트에서)
# @spaces.GPU를 _generate마다 붙이면 동일 요청 내 다중 GPU 할당 실패함
# =============================================================================

def _generate(input_ids, attention_mask, gen_params):
    """모델 추론 (GPU 컨텍스트는 호출자에서 관리)

    ZeroGPU는 동일 Gradio 요청 내 다중 @spaces.GPU 호출 시 두 번째부터 실패.
    app.py에서 전체 research를 @spaces.GPU(duration=300)으로 단일 래핑.
    """
    import torch

    device = next(_model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = _model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_params,
        )
    new_tokens = outputs[0][input_ids.shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)


# =============================================================================
# ZeroGPUClient
# =============================================================================
class ZeroGPUClient:
    """HuggingFace 모델 추론 클라이언트

    HF Spaces: @spaces.GPU + transformers (로컬 GPU, float16)
    로컬: HF Inference API (원격)
    """

    def __init__(self):
        if _ON_SPACES:
            self._backend = "spaces_local"
            logger.info(f"[ZeroGPU] Spaces 로컬 모드 ({MODEL_ID})")
        else:
            self._backend = "inference_api"
            self._client = self._init_inference_client()
            logger.info(f"[ZeroGPU] Inference API 모드 ({MODEL_ID})")

    def _init_inference_client(self):
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        return InferenceClient(model=MODEL_ID, token=token)

    def health(self) -> Dict:
        return {
            "status": "ready",
            "model": MODEL_ID,
            "backend": self._backend,
        }

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        if self._backend == "inference_api":
            return self._chat_inference_api(messages, max_tokens, temperature, stop)
        return self._chat_spaces_local(messages, max_tokens, temperature, stop)

    def _chat_inference_api(self, messages, max_tokens, temperature, stop) -> Dict:
        start_time = time.time()
        try:
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                stop=stop or [],
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            choice = response.choices[0]
            usage = response.usage
            content = (choice.message.content or "").strip()
            return {
                "success": True,
                "content": content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                "execution_time": elapsed_ms,
                "model": MODEL_ID,
            }
        except Exception as e:
            logger.error(f"Inference API 실패: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "execution_time": int((time.time() - start_time) * 1000),
            }

    @staticmethod
    def _postprocess(text: str, stop: Optional[List[str]] = None) -> str:
        """출력 후처리: stop sequence 절단 + 반복 패턴 제거"""
        # 1. Stop sequence에서 절단
        if stop:
            for seq in stop:
                idx = text.find(seq)
                if idx >= 0:
                    text = text[:idx]

        # 2. 반복 패턴 감지/절단 (같은 1-3자 패턴이 10회 이상 반복)
        repeat_match = re.search(r'(.{1,3})\1{9,}', text)
        if repeat_match:
            text = text[:repeat_match.start()]

        # 3. 콤마 구분 어구 반복 제거 ("X, X, X, X" → "X")
        text = re.sub(r'([\w가-힣]{2,15}(?:[·\s][\w가-힣]+)?)(?:[,，]\s*\1){2,}', r'\1', text)

        # 4. 중국어/일본어 구두점 반복 제거 (，、。等)
        text = re.sub(r'[，、。；：！？]{3,}', '', text)

        # 5. 문단/문장 단위 반복 제거 (20자+ 동일 블록이 3회 이상)
        para_repeat = re.search(r'(.{20,}?)\1{2,}', text, re.DOTALL)
        if para_repeat:
            text = text[:para_repeat.start() + len(para_repeat.group(1))]

        return text.strip()

    def _chat_spaces_local(self, messages, max_tokens, temperature, stop) -> Dict:
        start_time = time.time()
        try:
            input_text = _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = _tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"]  # CPU에서 준비, _generate 안에서 GPU로 이동
            attention_mask = inputs["attention_mask"]
            prompt_tokens = input_ids.shape[1]

            # MLX와 동일하게 최소 파라미터만 전달
            # top_k, top_p, repetition_penalty는 모델의 generation_config.json이 처리
            gen_params = {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "do_sample": temperature > 0,
            }

            text, completion_tokens = _generate(input_ids, attention_mask, gen_params)
            text = self._postprocess(text, stop)

            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "content": text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "execution_time": elapsed_ms,
                "model": MODEL_ID,
            }
        except Exception as e:
            logger.error(f"Spaces 로컬 추론 실패: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "execution_time": int((time.time() - start_time) * 1000),
            }

    def generate_report(
        self, system_prompt, user_prompt, max_tokens=2048, temperature=0.7
    ) -> Dict:
        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
