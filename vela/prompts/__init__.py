"""Research System Prompts

CoT 리서치 에이전트를 위한 시스템 프롬프트
"""

from .research import (
    RESEARCH_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT_TEMPLATE,
    RESEARCH_SYNTHESIS_PROMPT,
    get_research_prompt,
    get_research_system_prompt,
)

__all__ = [
    "RESEARCH_SYSTEM_PROMPT",
    "RESEARCH_SYSTEM_PROMPT_TEMPLATE",
    "RESEARCH_SYNTHESIS_PROMPT",
    "get_research_prompt",
    "get_research_system_prompt",
]
