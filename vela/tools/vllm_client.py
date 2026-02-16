"""vLLM 직접 연결 클라이언트

로컬 또는 원격 vLLM 서버에 직접 연결하는 클라이언트.
RunPod Serverless API가 아닌 표준 OpenAI 호환 API 사용.
"""

import os
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class VLLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "intrect/vela"
    timeout: int = 120


class VLLMClient:
    """vLLM 직접 연결 클라이언트"""

    def __init__(self, config: Optional[VLLMConfig] = None):
        if config:
            self.config = config
        else:
            self.config = VLLMConfig(
                base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                model_name=os.getenv("VLLM_MODEL_NAME", "intrect/vela"),
            )

    def health(self) -> Dict:
        """서버 상태 확인"""
        try:
            resp = requests.get(f"{self.config.base_url}/models", timeout=10)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                return {
                    "status": "healthy",
                    "models": [m.get("id") for m in models],
                }
            return {"status": "unhealthy", "error": resp.status_code}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> Dict:
        """Chat Completion 호출

        Returns:
            Dict with keys: success, content, usage
        """
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout,
            )

            if response.status_code != 200:
                return {
                    "error": f"HTTP {response.status_code}",
                    "detail": response.text[:500],
                }

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            return {
                "success": True,
                "content": content.strip(),
                "usage": usage,
            }

        except requests.Timeout:
            return {"error": "Timeout", "detail": "Request timed out"}
        except Exception as e:
            return {"error": str(e)}

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
