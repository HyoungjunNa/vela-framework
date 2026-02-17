"""뉴스-기업 벡터 유사도 필터 (HF Spaces CPU 모드)

KR-SBERT로 뉴스 제목을 임베딩한 뒤, 사전 계산된 기업 벡터와
코사인 유사도(dot product, L2 정규화)로 무관 뉴스 제거.

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
_DEFAULT_THRESHOLD = 0.25  # 코사인 유사도 임계값


class NewsVectorFilter:
    """뉴스-기업 코사인 유사도 필터 (CPU 전용)"""

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        vector_path: Optional[Path] = None,
    ):
        self.threshold = threshold
        self._pkl_path = vector_path or _LEAN_PKL
        self._model = None
        self._embeddings: Optional[np.ndarray] = None   # (N, 768)
        self._code_to_idx: dict[str, int] = {}
        self._code_to_name: dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # 지연 로딩 (첫 filter_sources 호출 시 초기화)
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        if self._loaded:
            return self._embeddings is not None

        self._loaded = True  # 중복 시도 방지

        # 벡터 pkl 로드
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
        except Exception as e:
            logger.warning(f"[VectorFilter] pkl 로드 실패: {e}")
            return False

        # KR-SBERT 모델 로드 (CPU)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_NAME, device="cpu")
            logger.info(f"[VectorFilter] KR-SBERT 로드 완료 (CPU): {_MODEL_NAME}")
        except Exception as e:
            logger.warning(f"[VectorFilter] KR-SBERT 로드 실패: {e}")
            return False

        return True

    # ------------------------------------------------------------------
    # 기업 벡터 조회
    # ------------------------------------------------------------------

    def _get_company_vector(self, stock_code: str) -> Optional[np.ndarray]:
        """stock_code → 정규화된 기업 벡터 (768,)"""
        idx = self._code_to_idx.get(stock_code)
        if idx is None:
            return None
        return self._embeddings[idx]  # 이미 L2 정규화됨

    # ------------------------------------------------------------------
    # 메인 필터
    # ------------------------------------------------------------------

    def filter_sources(
        self,
        sources: List[Source],
        stock_code: str,
    ) -> List[Source]:
        """코사인 유사도 기반 뉴스 필터링

        Args:
            sources: 수집된 Source 리스트
            stock_code: 6자리 종목코드

        Returns:
            유사도 >= threshold인 Source 리스트
            (로드 실패 / 벡터 없음 시 원본 반환)
        """
        if not sources:
            return sources

        if not self._load():
            return sources

        company_vec = self._get_company_vector(stock_code)
        if company_vec is None:
            logger.debug(f"[VectorFilter] 기업벡터 없음: {stock_code}")
            return sources

        # 뉴스 텍스트 추출 (제목 + snippet)
        texts = [f"{s.title} {s.snippet}"[:256] for s in sources]

        try:
            emb = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device="cpu",
            )
        except Exception as e:
            logger.warning(f"[VectorFilter] 임베딩 실패: {e}")
            return sources

        # 코사인 유사도 = dot product (양쪽 모두 정규화됨)
        sims = emb @ company_vec  # (M,)

        filtered, removed = [], 0
        for src, sim in zip(sources, sims):
            if float(sim) >= self.threshold:
                filtered.append(src)
            else:
                removed += 1
                logger.debug(
                    f"[VectorFilter] 제거 (sim={sim:.2f}): {src.title[:50]}"
                )

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
