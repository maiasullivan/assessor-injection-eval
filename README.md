# assessor-injection-eval

Can text inside a submission prompt-inject an LLM assessor that scores it, and does spotlighting stop it?

A small evaluation built with [Inspect](https://inspect.aisi.org.uk). An LLM grades student essays from ASAP 2.0 against a 1–6 holistic rubric, and we test whether text appended to an essay can inflate its score.

## Status

- **Phase 0 (setup): done.** Both models answer through Inspect (`scripts/hello.py`).
  - API: `anthropic/claude-haiku-5-5` (Sonnet: `anthropic/claude-sonnet-5-5`)
  - Local: `ollama/qwen3-4b-assessor`, built from `ollama/Modelfile` = `qwen3:4b-instruct-2507-q4_K_M` (Qwen3-4B-Instruct-2507, Q4_K_M, non-thinking) with `num_ctx 8192`. White-box work will use the bf16 HF weights `Qwen/Qwen3-4B-Instruct-2507`, so there is a quantisation mismatch.
  - Inspect 0.3.289. Inspect does not send `temperature` to Anthropic models, so API runs use default sampling with 3 epochs.
- **Phase 1 (dataset): done.** Switched from synthetic job applications to **ASAP 2.0** human-scored essays ("Exploring Venus" prompt only). `scripts/build_asap_sample.py` builds a 60-essay main set (20 weak, 20 middling, 20 strong by human score), a 20-essay dev set for prompt tuning, and a 200-essay length-matched probe set, all disjoint. Contamination checks (`scripts/leakage_check.py`) on Haiku 5.5, Sonnet 5.5 and Qwen3-4B found no recall of scores from essay IDs (ρ = 0.00) and no verbatim continuation (0% 8-gram overlap). The synthetic job-application data is kept in `data/synthetic/` for a possible second domain.
- **Phase 2 (attacks and defence): done.** `src/assessor_eval/attacks.py` defines three attacks (`direct` instruction, `fake_note` authority, and `keyword` rubric-language stuffing with no instruction), each with three wordings chosen per essay by a stable hash, and appended at the end or inserted in the middle. `src/assessor_eval/defences.py` implements spotlighting by datamarking (spaces become `^`, line breaks kept, wrapped in `<essay>` tags) with a generic system-prompt addendum. 12 unit tests pass (`uv run pytest`).
- **Phase 3 (Inspect task): built; at Checkpoint 3.** `src/assessor_eval/task.py` (`assessor(attack, defence, position, split)`) builds the system and user messages literally per sample, so essay IDs and human scores never reach the model (enforced by tests). The scorer parses `SCORE: n` and reports mean score and MAE; unparseable replies are kept as unscored. Clean dev-set (n = 20) QWK against human scores: Sonnet 5.5 0.62, Haiku 5.5 0.42, Qwen3-4B 0.27 (0.38 with spotlighting); no parse failures. 36 tests pass.
- **Phase 4 (run matrix): done.** 3 models × 4 attacks × 2 defences × 60 essays (Haiku 3 epochs, Sonnet 2, Qwen 1); 3,840 scored replies, 0 parse failures; API cost $6.67.
- **Phase 5 (analysis): at Checkpoint 5.** `scripts/analyse.py` → `results/`. Headlines: Qwen3-4B passes 75–85% of weak essays under instruction or fake-note injection; Sonnet 5.5 obeys one instruction wording (23% overall) but discloses it in its rationale; Haiku 5.5 is essentially immune. Spotlighting blocks all Claude successes but leaves Qwen at 55% under fake-note. Rubric-language stuffing slightly *lowers* scores.

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
