"""ZeroGPU Client - HuggingFace Spaces GPU 추론

HF Spaces 환경: @spaces.GPU + transformers 로컬 로드 (4-bit)
로컬 환경: HF Inference API 원격 추론 (GPU 불필요)

RunPodClient 호환 인터페이스:
    client = ZeroGPUClient()
    result = client.chat([{"role": "user", "content": "질문"}])
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_ON_SPACES = bool(os.environ.get("SPACE_ID"))


@dataclass
class ZeroGPUConfig:
    model_id: str = "intrect/VELA"
    max_tokens_default: int = 2048
    load_in_4bit: bool = True
    gpu_duration: int = 120


class ZeroGPUClient:
    """HuggingFace 모델 추론 클라이언트

    HF Spaces: @spaces.GPU + transformers (로컬 GPU)
    로컬: HF Inference API (원격)
    """

    def __init__(self, config: Optional[ZeroGPUConfig] = None):
        self.config = config or ZeroGPUConfig()

        if _ON_SPACES:
            self._backend = "spaces_local"
            self._model = None
            self._tokenizer = None
            self._loaded = False
            logger.info(f"HF Spaces 모드: ZeroGPU 추론 ({self.config.model_id})")
        else:
            self._backend = "inference_api"
            self._client = self._init_inference_client()
            logger.info(f"Inference API 모드: 원격 추론 ({self.config.model_id})")

    def _init_inference_client(self):
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        return InferenceClient(model=self.config.model_id, token=token)

    def health(self) -> Dict:
        return {
            "status": "ready",
            "model": self.config.model_id,
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
                "model": self.config.model_id,
            }
        except Exception as e:
            logger.error(f"Inference API 실패: {e}")
            return {"success": False, "error": str(e), "execution_time": int((time.time() - start_time) * 1000)}

    def _chat_spaces_local(self, messages, max_tokens, temperature, stop) -> Dict:
        start_time = time.time()
        try:
            self._ensure_loaded()

            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self._model.device)
            attention_mask = inputs["attention_mask"].to(self._model.device)
            prompt_tokens = input_ids.shape[1]

            gen_params = {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "do_sample": temperature > 0,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }

            text, completion_tokens = _generate(
                self._model, self._tokenizer, input_ids, attention_mask, gen_params,
            )

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
                "model": self.config.model_id,
            }
        except Exception as e:
            logger.error(f"Spaces 로컬 추론 실패: {e}")
            return {"success": False, "error": str(e), "execution_time": int((time.time() - start_time) * 1000)}

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"모델 로딩: {self.config.model_id}")
        start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True,
        )

        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                pass

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, **load_kwargs,
        )

        logger.info(f"모델 로딩 완료: {time.time() - start:.1f}초")
        self._loaded = True

    def generate_report(self, system_prompt, user_prompt, max_tokens=2048, temperature=0.7) -> Dict:
        return self.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )


# @spaces.GPU 데코레이터 (HF Spaces 전용)
try:
    import spaces

    def _do_generate(model, tokenizer, input_ids, attention_mask, gen_params):
        import torch
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_params)
        new_tokens = outputs[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)

    _generate = spaces.GPU(duration=120)(_do_generate)

except ImportError:
    def _generate(model, tokenizer, input_ids, attention_mask, gen_params):
        import torch
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_params)
        new_tokens = outputs[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)
