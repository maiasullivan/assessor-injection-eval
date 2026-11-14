# assessor-injection-eval

Can text inside a submission prompt-inject an LLM assessor that scores it, and does spotlighting stop it?

A small evaluation built with [Inspect](https://inspect.aisi.org.uk). An LLM grades student essays from ASAP 2.0 against a 1–6 holistic rubric, and we test whether text appended to an essay can inflate its score.

## Status

- **Phase 0 (setup): done.** Both models answer through Inspect (`scripts/hello.py`).
  - API: `anthropic/claude-haiku-5-5` (Sonnet: `anthropic/claude-sonnet-5-5`)
  - Local: `ollama/qwen3-4b-assessor`, built from `ollama/Modelfile` = `qwen3:4b-instruct-2507-q4_K_M` (Qwen3-4B-Instruct-2507, Q4_K_M, non-thinking) with `num_ctx 8192`. White-box work will use the bf16 HF weights `Qwen/Qwen3-4B-Instruct-2507`, so there is a quantisation mismatch.
  - Inspect 0.3.289. Inspect does not send `temperature` to Anthropic models, so API runs use default sampling with 3 epochs.
- **Phase 1 (dataset): done.** Switched from synthetic job applications to **ASAP 2.0** human-scored essays ("Exploring Venus" prompt only). `scripts/build_asap_sample.py` builds a 60-essay main set (20 weak, 20 middling, 20 strong by human score), a 20-essay dev set for prompt tuning, and a 200-essay length-matched probe set, all disjoint. Contamination checks (`scripts/leakage_check.py`) on Haiku 5.5, Sonnet 5.5 and Qwen3-4B found no recall of scores from essay IDs (ρ = 0.00) and no verbatim continuation (0% 8-gram overlap). The synthetic job-application data is kept in `data/synthetic/` for a possible second domain.

## Setup

```
uv sync --all-groups
cp .env.example .env          # add ANTHROPIC_API_KEY
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama create qwen3-4b-assessor -f ollama/Modelfile
# ASAP 2.0: download ASAP2_train_sourcetexts.csv (Kaggle) into data/, then
uv run python scripts/build_asap_sample.py
uv run inspect eval scripts/hello.py --model ollama/qwen3-4b-assessor
```

## Data and licence

Essays and human scores: **ASAP 2.0** (The Learning Agency Lab, 2024), drawn from the **PERSUADE 2.0** corpus (Crossley et al.), both CC BY 4.0. The rubric in `data/asap/rubric.md` is PERSUADE's source-based holistic rubric. Only essay IDs, text and scores are kept; demographic fields are dropped. The raw CSV is not redistributed here.
