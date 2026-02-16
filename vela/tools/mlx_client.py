"""VELA MLX Client - Apple Silicon 최적화 모델용 클라이언트

MLX 서버에 연결하여 VELA Fine-tuned 모델 사용
RunPodClient 호환 인터페이스 제공

사용법:
    client = VELAMLXClient()
    result = client.chat([{"role": "user", "content": "질문"}])
"""

import os
import time
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VELAMLXConfig:
    """VELA MLX 서버 설정"""

    base_url: str = "http://localhost:8081/v1"
    timeout: int = 180  # 3분 타임아웃
    max_tokens_default: int = 2048


class VELAMLXClient:
    """VELA Fine-tuned MLX 모델 클라이언트

    RunPodClient와 호환되는 인터페이스 제공
    model 파라미터 생략으로 서버에 로드된 모델 사용
    """

    def __init__(self, config: Optional[VELAMLXConfig] = None):
        if config:
            self.config = config
        else:
            self.config = VELAMLXConfig(
                base_url=os.getenv(
                    "VELA_MLX_BASE_URL", "http://localhost:8081/v1"
                ),
            )
        self.headers = {"Content-Type": "application/json"}

    def health(self) -> Dict:
        """서버 상태 확인"""
        try:
            resp = requests.get(
                f"{self.config.base_url}/models", headers=self.headers, timeout=10
            )
            if resp.status_code == 200:
                return {"status": "ready", "server": self.config.base_url}
            return {"status": "error", "code": resp.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        async_mode: bool = False,
        poll_interval: float = 2.0,
        max_wait: int = 180,
    ) -> Dict:
        """Chat Completion 호출 (RunPodClient 호환)

        Args:
            messages: OpenAI 형식 메시지 리스트
            max_tokens: 최대 생성 토큰
            temperature: 샘플링 온도
            stop: 정지 시퀀스 (선택)

        Returns:
            Dict with keys: success, content, usage, execution_time
        """
        start_time = time.time()

        payload = {
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
                headers=self.headers,
                timeout=self.config.timeout,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "detail": response.text[:500],
                }

            return self._parse_response(response.json(), elapsed_ms)

        except requests.Timeout:
            return {
                "success": False,
                "error": "Timeout",
                "detail": f"Request timed out after {self.config.timeout}s",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_response(self, data: Dict, elapsed_ms: int) -> Dict:
        """응답 파싱"""
        if "choices" not in data or len(data["choices"]) == 0:
            return {"success": False, "error": "No choices in response", "raw": data}

        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "")
        usage = data.get("usage", {})

        return {
            "success": True,
            "content": content.strip(),
            "usage": usage,
            "execution_time": elapsed_ms,
            "finish_reason": finish_reason,
        }

    def generate_report(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Dict:
        """보고서 생성 헬퍼 (RunPodClient 호환)"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
