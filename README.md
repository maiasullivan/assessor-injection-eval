# assessor-injection-eval

Can text inside a submission prompt-inject an LLM assessor that scores it, and does spotlighting stop it?

A small evaluation built with [Inspect](https://inspect.aisi.org.uk). All data is synthetic, and the agency and role are fictional.

## Status

- **Phase 0 (setup): done.** Both models answer through Inspect (`scripts/hello.py`).
  - API: `anthropic/claude-haiku-5-5` (Sonnet: `anthropic/claude-sonnet-5-5`)
  - Local: `ollama/qwen3-4b-assessor`, built from `ollama/Modelfile` = `qwen3:4b-instruct-2507-q4_K_M` (Qwen3-4B-Instruct-2507, Q4_K_M, non-thinking) with `num_ctx 8192`. White-box work will use the bf16 HF weights `Qwen/Qwen3-4B-Instruct-2507`, so there is a quantisation mismatch.
  - Inspect 0.3.289. Inspect does not send `temperature` to Anthropic models, so API runs use default sampling with 3 epochs.

## Setup

```
uv sync --all-groups
cp .env.example .env          # add ANTHROPIC_API_KEY
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama create qwen3-4b-assessor -f ollama/Modelfile
uv run inspect eval scripts/hello.py --model ollama/qwen3-4b-assessor
```
