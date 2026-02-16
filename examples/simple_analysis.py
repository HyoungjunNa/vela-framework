"""VELA Framework - 간단한 사용 예제

환경 설정 후 실행:
    cp .env.example .env
    # .env에 API 키 설정
    python examples/simple_analysis.py
"""

from dotenv import load_dotenv

load_dotenv()

from vela import ResearchAgent
from vela.schemas import ResearchOptions


def main():
    # 1. 에이전트 초기화 (MLX 백엔드 사용)
    agent = ResearchAgent(llm_backend="mlx")

    # 2. 리서치 옵션 설정
    options = ResearchOptions(
        max_iterations=3,          # 빠른 데모를 위해 3회
        extract_content=True,      # 웹페이지 본문 추출
        enable_verification=False, # 검증 비활성화 (Perplexity 키 불필요)
    )

    # 3. 리서치 실행
    result = agent.research(
        query="SK하이닉스 HBM 시장 전망",
        options=options,
    )

    # 4. 결과 확인
    print(f"=== 리서치 결과 ===")
    print(f"쿼리: {result.query}")
    print(f"신뢰도: {result.confidence:.0%}")
    print(f"소스 수: {len(result.sources)}개")
    print(f"소요 시간: {result.metadata.elapsed_seconds:.1f}초")
    print()

    # 핵심 발견
    print("=== 핵심 발견 ===")
    for i, finding in enumerate(result.key_findings or [], 1):
        print(f"  {i}. {finding}")
    print()

    # 결론 (앞부분)
    print("=== 결론 (요약) ===")
    if result.conclusion:
        print(result.conclusion[:500])
    print()

    # 소스 목록
    print("=== 소스 ===")
    for src in result.sources[:5]:
        print(f"  [{src.source_type}] {src.title[:60]}")

    # 5. JSON으로 저장 (선택)
    from pathlib import Path

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    saved = ResearchAgent.save_with_metadata(
        result=result,
        output_path=output_dir / "example_result.json",
    )
    print(f"\n저장 완료: {saved['result']}")


if __name__ == "__main__":
    main()
