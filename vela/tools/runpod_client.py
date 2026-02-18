"""RunPod Serverless vLLM 클라이언트

VELA 모델용 vLLM Chat Completions 클라이언트
RunPod Serverless API를 통해 HuggingFace Hub 모델 실행
"""

import os
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RunPodConfig:
    api_key: str
    endpoint_id: str
    model_name: str = "intrect/vela"
    timeout: int = 180


class RunPodClient:
    """RunPod Serverless vLLM 클라이언트

    요청 포맷: input.messages + input.sampling_params
    응답 포맷: output.choices[0].message.content
    """

    def __init__(self, config: Optional[RunPodConfig] = None):
        if config:
            self.config = config
        else:
            self.config = RunPodConfig(
                api_key=os.getenv("RUNPOD_API_KEY", ""),
                endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID", ""),
                model_name=os.getenv("MODEL_NAME", "intrect/vela"),
            )

        self.base_url = f"https://api.runpod.ai/v2/{self.config.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def health(self) -> Dict:
        """엔드포인트 상태 확인"""
        try:
            resp = requests.get(
                f"{self.base_url}/health", headers=self.headers, timeout=30
            )
            return (
                resp.json() if resp.status_code == 200 else {"error": resp.status_code}
            )
        except Exception as e:
            return {"error": str(e)}

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        async_mode: bool = True,
        poll_interval: float = 3.0,
        max_wait: int = 600,
    ) -> Dict:
        """Chat Completion 호출 (vLLM Serverless)

        Args:
            messages: OpenAI 포맷 메시지 리스트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            stop: 정지 토큰 리스트
            async_mode: True면 비동기 모드 (기본값, cold start 대비)
            poll_interval: 비동기 모드에서 폴링 간격 (초)
            max_wait: 비동기 모드에서 최대 대기 시간 (초, cold start 포함)
        """
        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            # vLLM은 generation_config.json을 자동 적용하지 않으므로 명시적 전달
            "top_k": 20,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
        }
        if stop:
            sampling_params["stop"] = stop

        payload = {
            "input": {
                "messages": messages,
                "sampling_params": sampling_params,
            }
        }

        try:
            if async_mode:
                return self._async_request(payload, poll_interval, max_wait)
            else:
                return self._sync_request(payload)

        except requests.Timeout:
            return {"error": "Timeout", "detail": "Request timed out"}
        except Exception as e:
            return {"error": str(e)}

    def _sync_request(self, payload: Dict) -> Dict:
        """동기 요청 (짧은 응답용)"""
        response = requests.post(
            f"{self.base_url}/runsync",
            json=payload,
            headers=self.headers,
            timeout=self.config.timeout,
        )

        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}",
                "detail": response.text[:500],
            }

        return self._parse_response(response.json())

    def _async_request(
        self, payload: Dict, poll_interval: float, max_wait: int
    ) -> Dict:
        """비동기 요청 (긴 응답용)"""
        import time

        # 1. 작업 제출
        response = requests.post(
            f"{self.base_url}/run", json=payload, headers=self.headers, timeout=30
        )

        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}",
                "detail": response.text[:500],
            }

        job_data = response.json()
        job_id = job_data.get("id")

        if not job_id:
            return {"error": "No job ID returned", "raw": job_data}

        # 2. 결과 폴링
        status_url = f"{self.base_url}/status/{job_id}"
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            status_resp = requests.get(status_url, headers=self.headers, timeout=30)
            status_data = status_resp.json()
            status = status_data.get("status")

            if status == "COMPLETED":
                return self._parse_response(status_data)
            elif status == "FAILED":
                return {"error": "Job failed", "raw": status_data}

        return {
            "error": "Timeout",
            "detail": f"Job {job_id} did not complete in {max_wait}s",
        }

    def _parse_response(self, data: Dict) -> Dict:
        """응답 파싱 (OpenAI Chat Completions 포맷)"""
        if "output" not in data:
            return {"status": data.get("status", "UNKNOWN"), "raw": data}

        output = data["output"]
        if isinstance(output, list) and len(output) > 0:
            output = output[0]

        if "choices" in output:
            choice = output["choices"][0]
            if "message" in choice:
                text = choice["message"].get("content", "")
            elif "tokens" in choice:
                # RunPod vLLM: tokens 배열 형식 (["response text"])
                tokens = choice["tokens"]
                text = "".join(tokens) if isinstance(tokens, list) else str(tokens)
            else:
                text = choice.get("text", "")

            usage = output.get("usage", {})

            return {
                "success": True,
                "content": text.strip(),
                "usage": {
                    # RunPod vLLM: input/output 또는 prompt_tokens/completion_tokens
                    "prompt_tokens": usage.get("prompt_tokens", usage.get("input", 0)),
                    "completion_tokens": usage.get("completion_tokens", usage.get("output", 0)),
                    "total_tokens": usage.get("total_tokens",
                                              usage.get("input", 0) + usage.get("output", 0)),
                },
                "execution_time": data.get("executionTime", 0),
                "delay_time": data.get("delayTime", 0),
                "worker_id": data.get("workerId", ""),
                "model": output.get("model", self.config.model_name),
            }
        elif "error" in output or (
            isinstance(output, dict) and output.get("object") == "error"
        ):
            return {"error": output.get("message", str(output))}

        return {"error": "Unknown response format", "raw": output}

    def generate_report(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict:
        """보고서 생성 헬퍼"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
