"""VELA Framework - 중앙집중 설정

모든 환경변수 및 설정값을 단일 파일에서 관리.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "output"


# =============================================================================
# LLM Backends
# =============================================================================

# RunPod Serverless
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
MODEL_NAME = os.getenv("MODEL_NAME", "intrect/vela")

# MLX Server (Apple Silicon)
MLX_BASE_URL = os.getenv("VELA_MLX_BASE_URL", "http://localhost:8081/v1")

# vLLM Direct
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "intrect/vela")

# LM Studio
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:3000/v1")
LMSTUDIO_MODEL_NAME = os.getenv("LMSTUDIO_MODEL_NAME", "intrect/vela")

# Perplexity (Adversary Agent)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")


# =============================================================================
# Search APIs
# =============================================================================

def get_naver_api_keys():
    """Naver API 키 목록 로드 (NAVER_CLIENT_ID_1..9)"""
    keys = []
    for i in range(1, 10):
        client_id = os.getenv(f"NAVER_CLIENT_ID_{i}")
        client_secret = os.getenv(f"NAVER_CLIENT_SECRET_{i}")
        if client_id and client_secret:
            keys.append((client_id, client_secret))
    return keys


# =============================================================================
# Agent Defaults
# =============================================================================

DEFAULT_LLM_BACKEND = os.getenv("VELA_LLM_BACKEND", "runpod")
DEFAULT_MAX_ITERATIONS = int(os.getenv("VELA_MAX_ITERATIONS", "5"))
DEFAULT_MIN_CONFIDENCE = float(os.getenv("VELA_MIN_CONFIDENCE", "0.8"))
DEFAULT_MAX_SOURCES = int(os.getenv("VELA_MAX_SOURCES", "20"))


# =============================================================================
# Stock Code Mapping (built-in fallback)
# =============================================================================

STOCK_CODE_MAP = {
    "삼성전자": "005930",
    "삼성": "005930",
    "SK하이닉스": "000660",
    "하이닉스": "000660",
    "현대차": "005380",
    "현대자동차": "005380",
    "LG에너지솔루션": "373220",
    "네이버": "035420",
    "NAVER": "035420",
    "카카오": "035720",
    "삼성SDI": "006400",
    "LG화학": "051910",
    "포스코홀딩스": "005490",
    "포스코": "005490",
    "POSCO홀딩스": "005490",
    "POSCO": "005490",
    "삼성바이오로직스": "207940",
    "셀트리온": "068270",
    "현대모비스": "012330",
    "기아": "000270",
    "기아차": "000270",
    "SK이노베이션": "096770",
    "KB금융": "105560",
    "신한지주": "055550",
    "삼성생명": "032830",
    "LG전자": "066570",
    "한화에어로스페이스": "012450",
}
