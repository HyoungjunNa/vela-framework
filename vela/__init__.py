"""VELA Framework - Domain-Specialized LLM Research Agent

A Chain-of-Thought research agent for Korean financial markets,
powered by a fine-tuned 7B model that scores 87.5/100 on domain benchmarks.

Quick Start:
    from vela import ResearchAgent

    agent = ResearchAgent(llm_backend="mlx")
    result = agent.research("SK하이닉스 HBM 시장 전망")
    print(result.to_markdown())
"""

__version__ = "1.0.0"

from .agent import ResearchAgent
from .schemas import (
    ActionType,
    ResearchOptions,
    ResearchRequest,
    ResearchResponse,
    ResearchResult,
    ReasoningStep,
    Source,
    SourceType,
)

__all__ = [
    "ResearchAgent",
    "ActionType",
    "ResearchOptions",
    "ResearchRequest",
    "ResearchResponse",
    "ResearchResult",
    "ReasoningStep",
    "Source",
    "SourceType",
]
