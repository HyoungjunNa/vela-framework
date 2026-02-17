"""뉴스-기업 벡터 유사도 필터 (KR-SBERT CPU 버전)

sentence-transformers를 CPU 전용으로 완전 지연 로드.
→ 모듈 임포트 시 CUDA probe 없음 → ZeroGPU worker_init 충돌 없음.

데이터: vela/data/company_vectors_lean.pkl
  - embeddings: (N, 768) float32, L2 정규화 완료
  - code_to_idx: stock_code -> row index
  - code_to_name: stock_code -> 회사명
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..schemas import Source

logger = logging.getLogger(__name__)

_LEAN_PKL = Path(__file__).parent.parent / "data" / "company_vectors_lean.pkl"
_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
_DEFAULT_THRESHOLD = 0.50  # 코사인 유사도 임계값 (KR-SBERT 기준)
_BATCH_SIZE = 32


class NewsVectorFilter:
    """뉴스-기업 코사인 유사도 필터 (KR-SBERT CPU 기반)"""

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        vector_path: Optional[Path] = None,
    ):
        self.threshold = threshold
        self._pkl_path = vector_path or _LEAN_PKL
        self._model = None          # SentenceTransformer (lazy)
        self._embeddings: Optional[np.ndarray] = None   # (N, 768)
        self._code_to_idx: dict[str, int] = {}
        self._code_to_name: dict[str, str] = {}
        self._pkl_loaded = False

    # ------------------------------------------------------------------
    # 지연 로딩
    # ------------------------------------------------------------------

    def _load_pkl(self) -> bool:
        if self._pkl_loaded:
            return self._embeddings is not None

        self._pkl_loaded = True
        if not self._pkl_path.exists():
            logger.warning(f"[VectorFilter] pkl 없음: {self._pkl_path}")
            return False

        try:
            with open(self._pkl_path, "rb") as f:
                data = pickle.load(f)
            self._embeddings = data["embeddings"]    # (N, 768) float32
            self._code_to_idx = data["code_to_idx"]
            self._code_to_name = data["code_to_name"]
            logger.info(
                f"[VectorFilter] 기업벡터 로드 완료: {self._embeddings.shape[0]}개 기업"
            )
            return True
        except Exception as e:
            logger.warning(f"[VectorFilter] pkl 로드 실패: {e}")
            return False

    def _get_model(self):
        """SentenceTransformer 지연 초기화 (CPU 전용, CUDA probe 없음)"""
        if self._model is not None:
            return self._model

        try:
            # CUDA를 완전히 배제하고 CPU에서만 실행
            # import는 이 메서드 호출 시점에 최초 실행 (모듈 로드 시 아님)
            import os
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_NAME, device="cpu")
            logger.info(f"[VectorFilter] KR-SBERT 로드 완료 (CPU): {_MODEL_NAME}")
            return self._model
        except Exception as e:
            logger.warning(f"[VectorFilter] 모델 로드 실패: {e}")
            return None

    # ------------------------------------------------------------------
    # 임베딩
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """텍스트 리스트 → L2 정규화된 임베딩 (N, 768)"""
        model = self._get_model()
        if model is None:
            return None

        try:
            emb = model.encode(
                texts,
                batch_size=_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return emb.astype(np.float32)
        except Exception as e:
            logger.warning(f"[VectorFilter] 임베딩 실패: {e}")
            return None

    # ------------------------------------------------------------------
    # 기업 벡터 조회
    # ------------------------------------------------------------------

    def _get_company_vector(self, stock_code: str) -> Optional[np.ndarray]:
        idx = self._code_to_idx.get(stock_code)
        if idx is None:
            return None
        return self._embeddings[idx]

    # ------------------------------------------------------------------
    # 메인 필터
    # ------------------------------------------------------------------

    def filter_sources(
        self,
        sources: List[Source],
        stock_code: str,
    ) -> List[Source]:
        """코사인 유사도 기반 뉴스 필터링

        Returns:
            유사도 >= threshold인 Source 리스트
            (로드 실패 / 임베딩 실패 시 원본 그대로 반환)
        """
        if not sources:
            return sources

        if not self._load_pkl():
            return sources

        company_vec = self._get_company_vector(stock_code)
        if company_vec is None:
            logger.debug(f"[VectorFilter] 기업벡터 없음: {stock_code}")
            return sources

        texts = [f"{s.title} {s.snippet}"[:256] for s in sources]
        emb = self._embed(texts)
        if emb is None:
            return sources

        sims = emb @ company_vec   # (M,) cosine similarity

        filtered, removed = [], 0
        for src, sim in zip(sources, sims):
            if float(sim) >= self.threshold:
                filtered.append(src)
            else:
                removed += 1
                logger.debug(f"[VectorFilter] 제거 sim={sim:.2f}: {src.title[:50]}")

        logger.info(
            f"[VectorFilter] {stock_code}: {len(sources)} → {len(filtered)} "
            f"(제거 {removed}, threshold={self.threshold})"
        )
        return filtered


# 싱글톤
_instance: Optional[NewsVectorFilter] = None


def get_vector_filter(threshold: float = _DEFAULT_THRESHOLD) -> NewsVectorFilter:
    global _instance
    if _instance is None:
        _instance = NewsVectorFilter(threshold=threshold)
    return _instance
