# VELA Methodology

## 1. Reasoning Trace (RT) Format

VELA uses a structured Markdown-based reasoning trace format designed for 7B parameter models. The format reduces cognitive load compared to JSON while maintaining parseability.

### Format Specification

```
**Step N**:
**Thought**: [Analysis of current state and decision rationale]
**Action**: search | analyze | conclude
**Query**: [Search query - only when Action=search]
**Confidence**: N%
```

### Action Types

| Action | Description | When Used |
|--------|-------------|-----------|
| `search` | Execute web search via Naver/DDG | Need more information |
| `analyze` | Extract findings from collected sources | Enough sources to analyze |
| `conclude` | Synthesize final report | Confidence >= 80% |

### Step Header Variants

The parser handles 4 header formats:
- `**Step N**:` (standard)
- `**Step N**` (no colon)
- `**Step N: title**` (with title)
- `**Step N - title**` (dash separator)

### Tool Call Preservation

JSON blocks containing `"tool"` + `"params"` keys are preserved verbatim (not converted to markdown):

```json
{"tool": "search_news", "params": {"query": "SK하이닉스 HBM"}}
```

### Quick Assessment Preservation

JSON blocks with `"category"` + `"sentiment"` keys are also preserved:

```json
{"category": "실적/재무", "sentiment": "positive"}
```

## 2. CoT Protocol

### TODO-Based Reasoning

Unlike simple sequential CoT, VELA generates a TODO list at Step 1 and tracks completion:

1. **Step 1 (Think)**: Generate TODO list of information needs
   - Each TODO has `id`, `task`, `priority` (critical/high/medium/low), `status`
2. **Step 2+ (Search/Analyze)**: Work through TODOs
   - Update status as items are completed
   - Generate new TODOs if gaps discovered
3. **Conclude**: When all critical TODOs are done and confidence >= 80%

### Loop Prevention

- Maximum consecutive searches: 2 (then forced to `analyze`)
- Duplicate query detection: skip previously searched queries
- Maximum iterations: configurable (default: 5)

### Confidence Gating

```
confidence < 40%  -> Must search (insufficient data)
40% <= conf < 80% -> Can search or analyze
confidence >= 80% -> Can conclude
```

The `should_continue()` function considers:
- Step confidence
- TODO completion rate (especially critical items)
- Maximum iteration count

## 3. Training Data Pipeline

### SFT Data (58,206 samples)

| Source | Count | Method |
|--------|-------|--------|
| Haiku Batch 1-4 | ~20,000 | Claude Haiku API batch generation |
| Qwen ChatML | ~5,000 | Format conversion |
| Securities Reports | ~5,126 | PDF parsing + RT generation |
| Tool Calling | ~5,000 | Function call format conversion |
| Multi-turn 2T | 4,000 | 2-turn dialogue generation |
| Multi-turn 4T | 4,000 | 4-turn dialogue generation |
| Gap Fill (12 categories) | ~12,600 | Sonar + OpenAI Batch API |
| Others | ~2,480 | Labeled data, batch5 fallback |

### DPO Data (26,421 pairs)

| Source | Pairs | Rejection Type |
|--------|-------|----------------|
| DPO Dedup | 12,000 | Various quality issues |
| Multilingual Aug | 5,997 | Language mixing (CN/EN leak) |
| VELA ChatML | 5,000 | Qwen response quality |
| Batch5 Insufficient | 1,642 | Insufficient analysis |
| Chinese Leak v2 | 1,216 | Chinese character correction |
| Reasoning Trace 2K | 566 | RT format errors |

### DPO Rejection Categories

```
english_leak (30%): English terms inserted in Korean context
chinese_leak (30%): Chinese characters from Qwen base model
format_error (20%): Broken RT JSON/Markdown structure
short_response (20%): Insufficient analysis depth
```

### RT Format Migration

Original training data used JSON RT format. Converted to Markdown for reduced token count and improved 7B model parsing:

- **Before**: `{"thought": "...", "action": "search", "query": "..."}`
- **After**: `**Thought**: ...\n**Action**: search\n**Query**: ...`
- **Result**: 100% parse success rate across 101,296 RT blocks

## 4. Model Architecture

### Base
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Why Qwen**: Best multilingual base for CJK at 7B scale

### SFT Stage
- **LoRA Config**: r=64, alpha=128
- **Learning Rate**: 2e-4
- **Batch Size**: 4 (gradient accumulation 8)
- **Epochs**: ~3 on 58K samples
- **Output**: Merged BF16 weights (15GB)

### DPO Stage
- **LoRA Config**: r=16, alpha=32
- **Beta**: 0.1
- **Learning Rate**: 5e-5
- **Output**: LoRA adapter (155MB)
- **Key Goal**: Eliminate Chinese/English language leaks while preserving reasoning quality

### Quantization
- **MLX INT4**: 4GB, 16 tok/s on M1 Max (3.2x faster than BF16 CPU)
- **GGUF Q4_K_M**: ~4.4GB for llama.cpp compatibility

## 5. Search Architecture

### Multi-Source Strategy

```
Query -> [Naver News API] -> results (Korean news, high relevance)
      -> [DuckDuckGo]     -> results (international, fallback)
      -> Merge + Deduplicate -> Top-K sources
```

### Naver API Pool

4 parallel API key pools for rate limit management:
- `NAVER_CLIENT_ID_0` / `NAVER_CLIENT_SECRET_0`
- `NAVER_CLIENT_ID_1` / `NAVER_CLIENT_SECRET_1`
- etc.

Round-robin selection with automatic failover.

### Stock Code Resolution

Built-in mapping for major Korean stocks (KOSPI/KOSDAQ top 25):
```python
STOCK_CODE_MAP = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    ...
}
```

Extensible via configuration.

## 6. Adversary Verification

Optional cross-verification using Perplexity Sonar API:

1. **Input**: ResearchResult (conclusion + sources + key findings)
2. **Process**: Independent fact-checking against Perplexity's real-time search
3. **Output**: VerificationResult with:
   - `verdict`: ACCEPT / REVISE / NEED_MORE_SEARCH
   - `issues`: List of factual concerns
   - `suggested_counter_queries`: Additional searches recommended
   - `confidence`: Verification confidence score

### Fail-Open Prevention

Verification failures are explicitly recorded (not silently accepted):
- Failed verification -> `verdict=REVISE` with error details
- No silent `ACCEPT` on API errors

## 7. Benchmark Methodology

### Evaluation Framework

10 Korean financial domain prompts across 6 dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Domain Knowledge | 25% | Korean market terminology, regulations |
| Reasoning Quality | 20% | Logical analysis depth |
| Data Accuracy | 20% | Factual correctness |
| Korean Fluency | 15% | Natural Korean, no language mixing |
| Actionability | 10% | Practical investment insights |
| Structure | 10% | Report organization |

### Scoring

- 100-point scale per prompt
- 4-model comparison: VELA, Qwen base, GPT-4o, Exaone 3.5
- Human evaluation + GPT-4o automated scoring
- Final score: weighted average across all prompts

### Results Summary

| Model | Score | Key Strength | Key Weakness |
|-------|-------|--------------|--------------|
| **VELA 7B** | **87.5** | Domain depth, structured RT | Smaller context window |
| GPT-4o | 81.0 | General reasoning | Korean financial terminology |
| Exaone 3.5 | 74.5 | Korean fluency | Shallow domain analysis |
| Qwen 2.5 7B | 72.0 | Multilingual base | Chinese/English leaks |
