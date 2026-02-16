# VELA Framework

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/intrect/vela-demo)

**Domain-Specialized LLM Research Agent for Korean Financial Markets**

VELA is an open-source research agent framework that demonstrates how a single developer can build a domain-specialized LLM system competitive with $100M+ proprietary projects -- for under $235/month in compute costs.

## Key Results

| Metric | VELA (7B) | Qwen 2.5 7B Base | GPT-4o | Exaone 3.5 7.8B |
|--------|-----------|-------------------|--------|------------------|
| Domain Knowledge (100pt) | **87.5** | 72.0 | 81.0 | 74.5 |
| Korean Fluency | Native | Mixed (CN leak) | Good | Native |
| Reasoning Trace | Structured | None | Free-form | None |

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Training**: SFT (58K samples) + DPO (26K pairs) on Korean financial domain
- **Inference**: 16 tok/s on Apple Silicon (MLX 4-bit), RunPod Serverless, or any vLLM server
- **License**: MIT

## Architecture

```
User Query
    |
    v
[ResearchAgent] -- CoT Reasoning Loop (Think -> Search -> Analyze -> Conclude)
    |
    +-- [CoTReasoningEngine] -- TODO-based iterative reasoning with confidence gating
    +-- [ResearchSearchModule] -- Multi-source web search (Naver + DuckDuckGo)
    +-- [ContentExtractor] -- Web page & PDF content extraction
    +-- [AdversaryAgent] -- Cross-verification via Perplexity API (optional)
    |
    v
ResearchResult (structured JSON with trajectory, claim-evidence mapping)
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/intrect/vela-framework.git
cd vela-framework

# 2. Install
pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env with your API keys (RUNPOD_API_KEY, NAVER_CLIENT_ID, etc.)

# 4. Run
python inference.py --query "SK하이닉스 HBM 시장 전망" --backend mlx
```

## Installation

### Requirements

- Python 3.10+
- At least one LLM backend configured (RunPod, MLX, or vLLM)

### From Source

```bash
pip install -e .
```

### Dependencies

Core (auto-installed):
- `pydantic>=2.0` -- Structured schemas
- `requests` -- HTTP client
- `python-dotenv` -- Environment configuration
- `duckduckgo-search` -- Web search fallback
- `beautifulsoup4` -- Content extraction

## Configuration

All configuration is via environment variables. Copy `.env.example` and fill in your keys:

```bash
# LLM Backends (configure at least one)
RUNPOD_API_KEY=your_key          # RunPod Serverless
RUNPOD_ENDPOINT_ID=your_endpoint
VELA_MLX_BASE_URL=http://localhost:8081/v1   # MLX server
VLLM_BASE_URL=http://localhost:8000/v1       # vLLM server

# Search APIs
NAVER_CLIENT_ID_0=your_id       # Naver Search API
NAVER_CLIENT_SECRET_0=your_secret

# Verification (optional)
PERPLEXITY_API_KEY=your_key     # Adversary Agent
```

## Usage

### Python API

```python
from vela import ResearchAgent
from vela.schemas import ResearchOptions

# Initialize with your preferred backend
agent = ResearchAgent(llm_backend="mlx")

# Run research
result = agent.research(
    query="SK하이닉스 HBM 시장 전망",
    options=ResearchOptions(max_iterations=5),
)

# Access results
print(result.conclusion)
print(f"Confidence: {result.confidence:.0%}")
print(f"Sources: {len(result.sources)}")

# Save with full metadata (for training data generation)
from pathlib import Path
ResearchAgent.save_with_metadata(result, Path("output/result.json"))
```

### CLI

```bash
# Basic research
python inference.py -q "삼성전자 반도체 전략" -b mlx

# With verification
python inference.py -q "네이버 AI 전략" --verify

# Save output
python inference.py -q "카카오 실적" -o result.json

# Verbose logging
python inference.py -q "현대차 전기차" -v
```

### Web Interface

Try VELA directly in your browser via Gradio:

```bash
# Install with web dependencies
pip install -e ".[web]"

# Launch local demo
python app.py
# Opens at http://localhost:7860
```

Or try the hosted demo on [HuggingFace Spaces](https://huggingface.co/spaces/intrect/vela-demo).

### LLM Backends

| Backend | Use Case | Setup |
|---------|----------|-------|
| `runpod` | Cloud GPU inference | Set `RUNPOD_API_KEY` + `RUNPOD_ENDPOINT_ID` |
| `mlx` | Apple Silicon local | Run MLX server, set `VELA_MLX_BASE_URL` |
| `vllm` | Any GPU server | Run vLLM, set `VLLM_BASE_URL` |

## How It Works

### Chain-of-Thought Reasoning

VELA uses a TODO-based CoT protocol where each research iteration follows:

1. **Think**: Analyze current state and generate a TODO list
2. **Search**: Execute web searches (Naver + DuckDuckGo)
3. **Analyze**: Extract intermediate findings from collected sources
4. **Conclude**: Synthesize final report when confidence threshold is met

### Reasoning Trace Format (Markdown)

```
**Step 1**:
**Thought**: SK하이닉스의 HBM 시장 점유율과 경쟁 구도 분석 필요
**Action**: search
**Query**: SK하이닉스 HBM3E 시장점유율 2025
**Confidence**: 35%

**Step 2**:
**Thought**: HBM 매출 비중과 영업이익률 데이터 확보 완료
**Action**: analyze
**Confidence**: 65%

**Step 3**:
**Thought**: 충분한 데이터 수집, 결론 도출 가능
**Action**: conclude
**Confidence**: 85%
```

### Confidence Gating

The system uses a confidence gate at multiple levels:
- **Per-step**: Each reasoning step reports confidence (0-100%)
- **Continuation**: Research continues until confidence >= 80% or max iterations
- **Synthesis**: Final report includes overall confidence score

### Adversary Verification (Optional)

When `--verify` is enabled, an independent AdversaryAgent cross-checks the research output using the Perplexity API, identifying:
- Factual inconsistencies
- Unsupported claims
- Missing counter-arguments

## Training Your Own Model

VELA's training pipeline produced the domain-specialized model through:

1. **SFT** (Supervised Fine-Tuning): 58K samples of Korean financial analysis with structured reasoning traces
2. **DPO** (Direct Preference Optimization): 26K pairs targeting language purity (eliminating Chinese/English leaks from Qwen base) and reasoning quality

### Model Weights

- **HuggingFace**: [intrect/vela](https://huggingface.co/intrect/vela) (GGUF Q4_K_M)
- **Base**: Qwen/Qwen2.5-7B-Instruct + LoRA (r=64, alpha=128)

### Training Cost Breakdown

| Component | Cost | Notes |
|-----------|------|-------|
| RunPod RTX 4090 | ~$50/month | SFT + DPO training |
| Haiku API (data gen) | ~$80 | 5 batches, 50K samples |
| Naver/Search APIs | ~$30/month | Data collection |
| Perplexity API | ~$20/month | Adversary verification |
| **Total** | **~$235/month** | |

## Project Structure

```
vela-framework/
├── app.py               # Gradio web demo (HF Spaces)
├── inference.py          # CLI entry point
├── vela/                 # Core package
│   ├── agent.py          # ResearchAgent orchestrator
│   ├── reasoning.py      # CoT reasoning engine
│   ├── search.py         # Multi-source web search
│   ├── schemas.py        # Pydantic data models
│   ├── content_extractor.py
│   ├── adversary.py      # Verification agent
│   ├── config.py         # Centralized configuration
│   ├── prompts/          # System & research prompts
│   └── tools/            # LLM clients & utilities
│       ├── runpod_client.py
│       ├── mlx_client.py
│       ├── vllm_client.py
│       ├── ddg_search.py
│       ├── naver_search.py
│       ├── confidence_gate.py
│       └── fact_extractor.py
├── docs/
│   └── METHODOLOGY.md    # Detailed methodology
└── examples/
    └── simple_analysis.py
```

## Methodology

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for detailed documentation on:
- Reasoning Trace format specification
- CoT protocol design
- Training data generation pipeline
- DPO strategy for language purity
- Benchmark methodology

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)

## Citation

```bibtex
@software{vela_framework_2026,
  title={VELA Framework: Domain-Specialized LLM Research Agent for Korean Financial Markets},
  author={intrect},
  year={2026},
  url={https://github.com/intrect/vela-framework}
}
```
