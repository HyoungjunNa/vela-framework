#!/usr/bin/env python3
"""VELA Framework - 추론 데모

사용법:
    # RunPod 백엔드 (기본)
    python inference.py --query "SK하이닉스 HBM 시장 전망"

    # MLX 백엔드 (Apple Silicon)
    python inference.py --query "삼성전자 분석" --backend mlx

    # vLLM 백엔드
    python inference.py --query "네이버 AI 전략" --backend vllm

    # Adversary 검증 포함
    python inference.py --query "SK하이닉스 HBM" --verify

    # JSON 파일로 저장
    python inference.py --query "삼성전자 반도체" --output result.json
"""

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from vela import ResearchAgent
from vela.schemas import ResearchOptions


def main():
    parser = argparse.ArgumentParser(
        description="VELA Research Agent - Korean Financial Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python inference.py -q "SK하이닉스 HBM 시장 전망"
  python inference.py -q "삼성전자 분석" -b mlx
  python inference.py -q "네이버 AI" -b vllm --verify
  python inference.py -q "카카오 실적" -o result.json --markdown
        """,
    )
    parser.add_argument(
        "--query", "-q", type=str, required=True, help="리서치 쿼리",
    )
    parser.add_argument(
        "--backend", "-b", type=str, default="runpod",
        choices=["runpod", "mlx", "vllm"],
        help="LLM 백엔드 (기본: runpod)",
    )
    parser.add_argument(
        "--max-iterations", "-i", type=int, default=5,
        help="최대 추론 반복 횟수 (기본: 5)",
    )
    parser.add_argument(
        "--no-content", action="store_true",
        help="콘텐츠 본문 추출 비활성화 (속도 향상)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Adversary Agent 검증 활성화 (Perplexity API 필요)",
    )
    parser.add_argument(
        "--verify-model", type=str, default="sonar",
        help="검증 Perplexity 모델 (기본: sonar)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="결과 저장 경로 (JSON)",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="마크다운 형식으로 출력",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="상세 로깅",
    )
    parser.add_argument(
        "--stock-code", type=str, help="종목코드 (예: 005930)",
    )

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 에이전트 초기화
    print(f"\n[VELA] 백엔드: {args.backend}")
    print(f"[VELA] 리서치 시작: {args.query}\n")

    agent = ResearchAgent(
        llm_backend=args.backend,
        extract_content=not args.no_content,
    )

    options = ResearchOptions(
        max_iterations=args.max_iterations,
        extract_content=not args.no_content,
        enable_verification=args.verify,
        verification_model=args.verify_model,
    )

    # 리서치 실행
    result = agent.research(
        query=args.query,
        options=options,
        stock_code=args.stock_code,
    )

    # 결과 출력
    if args.markdown:
        print(result.to_markdown())
    else:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    # 파일 저장
    if args.output:
        from pathlib import Path

        saved = ResearchAgent.save_with_metadata(
            result=result,
            output_path=Path(args.output),
            save_markdown=args.markdown,
        )
        print(f"\n[VELA] 결과 저장: {saved['result']}")
        print(f"[VELA] 메타데이터 저장: {saved['metadata']}")

    # 요약
    print(f"\n{'='*50}")
    print(f"[VELA] 리서치 완료")
    print(f"  신뢰도: {result.confidence:.0%}")
    print(f"  소스: {len(result.sources)}개")
    print(f"  반복: {result.metadata.iterations}회")
    print(f"  소요: {result.metadata.elapsed_seconds:.1f}초")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
