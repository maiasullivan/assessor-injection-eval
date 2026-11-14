"""White-box Part A: Qwen3-4B's probability over scores 1-6, per essay x condition.

    uv run --group whitebox python whitebox/score_distribution.py            # main matrix + wording follow-up
    uv run --group whitebox python whitebox/score_distribution.py --limit 3  # smoke test

Prompts are rebuilt with the Inspect task's own build_sample (same system prompt, injection and
spotlighting), put through Qwen's chat template, and prefilled with the assistant's "SCORE:" up
to the digit. One forward pass reads the next-token distribution over the six digit tokens.

The system prompt is identical within a defence, so its KV cache is computed once and cropped back
after each sample. --check compares that against a full forward pass for a few samples.

Precision: float16, not bfloat16. On close calls (e.g. 3 vs 4) bf16 rounding moved digit
probabilities by up to ~0.1 between two mathematically equivalent passes; fp16 agreed to ~0.015.
float32 (16 GB) does not fit alongside the OS on a 16 GB M1.

Writes results/whitebox/score_dist.csv: one row per (set, attack, defence, variant, essay),
appended as it goes; a re-run resumes where it stopped.
"""

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from assessor_eval.attacks import ATTACKS
from assessor_eval.task import DEFENCES, build_sample, load_split

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUT = Path(__file__).resolve().parent.parent / "results" / "whitebox"
SCORES = range(1, 7)


def conditions(limit: int | None) -> list[tuple[str, dict, str, str, int | None]]:
    """(set, essay, attack, defence, forced variant): the main matrix, then the wording follow-up."""
    essays = load_split("main")[:limit]
    weak = [e for e in essays if e["tier"] == "weak"]
    main = [("main", e, a, d, None) for d in DEFENCES for a in ATTACKS for e in essays]
    variants = [("variants", e, a, d, v) for d in DEFENCES for a in ("direct", "fake_note")
                for v in range(len(ATTACKS[a])) for e in weak]
    return main + variants


def score_tokens(tok) -> tuple[str, list[int]]:
    """The assistant prefill up to the digit, and the token id of each digit 1-6 that follows it.

    Asserts each "SCORE: n" tokenizes as the same prefix plus one digit token, so the prefill is
    exactly how the model's own reply would be tokenized.
    """
    seqs = [tok.encode(f"SCORE: {s}", add_special_tokens=False) for s in SCORES]
    prefix = seqs[0][:-1]
    assert all(s[:-1] == prefix for s in seqs), "digits do not tokenize as one final token"
    return tok.decode(prefix), [s[-1] for s in seqs]


def prompt_ids(tok, sample, prefill: str) -> tuple[list[int], list[int]]:
    """Token ids of the system-only prefix and of the full prompt (system, user, assistant prefill)."""
    msgs = [{"role": m.role, "content": m.text} for m in sample.input]
    system = tok.apply_chat_template(msgs[:1], tokenize=False)
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    sys_ids, full_ids = tok.encode(system, add_special_tokens=False), tok.encode(full, add_special_tokens=False)
    assert full_ids[: len(sys_ids)] == sys_ids, "system prefix does not tokenize identically"
    return sys_ids, full_ids


@torch.no_grad()
def digit_logprobs(model, ids: list[int], digits: list[int], cache=None, start: int = 0) -> torch.Tensor:
    """Full-vocab log-probs of the six digit tokens at the next position. With a cache holding ids[:start]."""
    x = torch.tensor([ids[start:]], device=model.device)
    out = model(input_ids=x, past_key_values=cache, use_cache=cache is not None, logits_to_keep=1)
    return torch.log_softmax(out.logits[0, -1].float(), dim=-1)[digits].cpu()


def summarise(logp: torch.Tensor) -> dict:
    p = torch.softmax(logp, dim=0)  # renormalised over the six digits
    row = {f"p{s}": float(p[i]) for i, s in enumerate(SCORES)}
    row.update(
        digit_mass=float(logp.exp().sum()),  # how much of the full distribution the six digits get
        argmax=int(p.argmax()) + 1,
        expected=float(sum(s * p[i] for i, s in enumerate(SCORES))),
        p_pass=float(p[3:].sum()),  # p(score >= 4)
        entropy=float(-(p * p.clamp_min(1e-12).log()).sum() / math.log(2)),  # bits; max log2(6) = 2.58
        max_prob=float(p.max()),
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="first N main-set essays only (smoke test)")
    parser.add_argument("--check", type=int, default=3, help="samples to verify against a full forward pass")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    prefill, digits = score_tokens(tok)
    print(f"prefill {prefill!r}, digit tokens {digits}, device {device}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("score_dist.csv" if args.limit is None else "score_dist_smoke.csv")
    done = set()
    if path.exists():  # resume: skip rows already written
        prev = pd.read_csv(path)
        done = set(zip(prev.set, prev.attack, prev.defence, prev.variant, prev.essay_id))
    conds = conditions(args.limit)
    print(f"{len(conds)} conditions, {len(done)} already done", flush=True)

    cache_for, cache, sys_len = None, None, 0
    t0, n_new = time.time(), 0
    for i, (set_, essay, attack, defence, variant) in enumerate(conds):
        sample = build_sample(essay, attack, defence, "end", variant)
        if (set_, attack, defence, sample.metadata["variant"], essay["essay_id"]) in done:
            continue
        sys_ids, ids = prompt_ids(tok, sample, prefill)
        if cache_for != defence:  # one system-prompt cache at a time (conditions are ordered by defence)
            cache = None
            with torch.no_grad():
                cache = model(input_ids=torch.tensor([sys_ids], device=device), use_cache=True, logits_to_keep=1).past_key_values
            cache_for, sys_len = defence, len(sys_ids)
        assert len(sys_ids) == sys_len, "system prompt differs within a defence"
        logp = digit_logprobs(model, ids, digits, cache, start=sys_len)
        cache.crop(-(cache.get_seq_length() - sys_len))  # drop this sample's tokens
        if n_new < args.check:
            full = digit_logprobs(model, ids, digits)
            diff = (full.exp() - logp.exp()).abs().max().item()
            print(f"cache check {i}: max |dp| = {diff:.4f}", flush=True)
            assert diff < 0.03, "cached prefix gives different probabilities from a full pass"
        row = {"set": set_, "essay_id": essay["essay_id"], "tier": essay["tier"], "true": essay["true_score"],
               "attack": attack, "defence": defence, "variant": sample.metadata["variant"],
               "prompt_tokens": len(ids), **summarise(logp)}
        pd.DataFrame([row]).round(5).to_csv(path, mode="a", header=not path.exists(), index=False)
        n_new += 1
        if device == "mps":
            torch.mps.empty_cache()  # without this the allocator's cache grows until the machine swaps
        if n_new % 10 == 0:
            print(f"{i + 1}/{len(conds)}  {(time.time() - t0) / n_new:.1f}s per row", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
