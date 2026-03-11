# CLAUDE.md

## What This Is

credence-router: transparent, cost-optimal tool routing via EU maximisation.
Replaces LangChain/LangGraph's opaque LLM-based routing with principled VOI
calculations. Zero routing cost, <1ms routing latency, learns from feedback.

## Development Commands

```bash
uv sync --all-groups
pytest tests/
ruff check src/ tests/
ruff format src/ tests/

# Simulated benchmark (no API keys needed)
credence-router bench --run --simulate

# Full benchmark (needs ANTHROPIC_API_KEY, PERPLEXITY_API_KEY)
credence-router bench --run

# Interactive routing
credence-router route "What is 2+2?" -o "3" "4" "5" "6"
```

## Architecture

- `router.py` — Router wraps credence's BayesianAgent, orchestrates tool selection
- `tool.py` — Tool protocol: name, cost, latency, query(), coverage()
- `answer.py` — Answer dataclass with full reasoning trace
- `tools/` — Built-in tools: calculator, claude, perplexity, simulated
- `baselines/` — Comparison agents: random, always-cheapest, always-best, langgraph
- `benchmark.py` — Benchmark runner with cost + latency tracking
- `categories.py` — Reuses credence's 5 categories + keyword inference
- `questions.py` — Reuses credence's 50-question bank
- `cli.py` — CLI entry points: `bench` and `route` subcommands

## Design Principles

Same as credence: everything is EU maximisation, no hacks, LLM outputs are data.

## Dependencies

- `credence-agents` — Bayesian inference engine
- `numpy` — numerical computation
- `anthropic` — Claude API (optional)
- `httpx` — Perplexity API (optional)
- `langgraph` + `langchain-anthropic` — baseline comparison (optional)

Ruff: line-length = 99.
