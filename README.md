# assessor-injection-eval

Can text inside a submission prompt-inject an LLM assessor that scores it, and does spotlighting stop it?

A small evaluation built with [Inspect](https://inspect.aisi.org.uk). All data is synthetic, and the agency and role are fictional.

## Status

- **Phase 0 (setup): done.** Both models answer through Inspect (`scripts/hello.py`).
  - API: `anthropic/claude-haiku-5-5` (Sonnet: `anthropic/claude-sonnet-5-5`)
  - Local: `ollama/qwen3-4b-assessor`, built from `ollama/Modelfile` = `qwen3:4b-instruct-2507-q4_K_M` (Qwen3-4B-Instruct-2507, Q4_K_M, non-thinking) with `num_ctx 8192`. White-box work will use the bf16 HF weights `Qwen/Qwen3-4B-Instruct-2507`, so there is a quantisation mismatch.
  - Inspect 0.3.289. Inspect does not send `temperature` to Anthropic models, so API runs use default sampling with 3 epochs.
- **Phase 1 (dataset): generated, awaiting blind labels.** `data/role.md` describes a fictional Data Analyst role at Example Co. `scripts/generate_applications.py` (Claude Sonnet 5.5) wrote 30 main-set applications (10 per target tier, 295–487 words) and 150 probe-set applications (75 weak, 75 strong). Every application passed the format checks: no keyword lists, no injection-like text, no real organisations. Labelling is blind (`data/to_label.md` → `data/labels.csv` → `scripts/merge_labels.py`).

## Setup

```
uv sync --all-groups
cp .env.example .env          # add ANTHROPIC_API_KEY
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama create qwen3-4b-assessor -f ollama/Modelfile
uv run inspect eval scripts/hello.py --model ollama/qwen3-4b-assessor
```
