"""ZeroGPU Client - HuggingFace Spaces GPU 추론

HF Spaces 환경: @spaces.GPU + transformers 모듈 레벨 로드 (float16)
로컬 환경: HF Inference API 원격 추론 (GPU 불필요)

RunPodClient 호환 인터페이스:
    client = ZeroGPUClient()
    result = client.chat([{"role": "user", "content": "질문"}])
"""

import logging
import os
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
# @spaces.GPU 데코레이터 (HF Spaces 전용)
# =============================================================================
if _has_spaces:

    @spaces.GPU(duration=120)
    def _generate(input_ids, attention_mask, gen_params):
        import torch

        # ZeroGPU가 모델을 CUDA로 옮긴 후 입력도 같은 디바이스로 이동
        device = next(_model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = _model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_params,
            )
        new_tokens = outputs[0][input_ids.shape[1] :]
        return _tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)

else:

    def _generate(input_ids, attention_mask, gen_params):
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
        new_tokens = outputs[0][input_ids.shape[1] :]
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
                top_p=0.9,
                stop=stop or [],
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            choice = response.choices[0]
            usage = response.usage
            return {
                "success": True,
                "content": (choice.message.content or "").strip(),
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

            gen_params = {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "do_sample": temperature > 0,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }

            text, completion_tokens = _generate(input_ids, attention_mask, gen_params)

            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "content": text.strip(),
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
