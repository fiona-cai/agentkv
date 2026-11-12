"""Benchmark: single-shot compression vs repeated (agent-style) compaction.

single_shot : read the whole episode + the questions, compress once, answer.
              This is how most KV-compression papers evaluate.
repeated    : compress after every agent turn (the cache is capped at the
              budget the whole time), then the questions arrive at the end.
              This is how an agent actually has to run.

Usage:  python -m agentkv.bench --episodes 200
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .cache import KVCache, bytes_per_token_ratio
from .episodes import (EpisodeConfig, Vocab, Segment, make_episode,
                       USER, TYPE_NAMES, DV)
from .policies import POLICIES

MEM_FRACS = [0.02, 0.04, 0.06, 0.10, 0.15, 0.25, 0.40]

# (label, policy name, bits)
RUNS = [
    ("Recent window", "recent", 16),
    ("StreamingLLM", "streaming", 16),
    ("H2O", "h2o", 16),
    ("SnapKV", "snapkv", 16),
    ("AgentKV: type only", "agent_type", 16),
    ("AgentKV: addressability only", "agent_addr", 16),
    ("AgentKV", "agentkv", 16),
    ("AgentKV + 4-bit", "agentkv", 4),
]


def question_segment(ep, vocab, rng):
    keys = vocab.key_emb[ep.fact_keys[ep.questions]]
    m = len(keys)
    k = 0.2 * rng.standard_normal((m, keys.shape[1]))
    v = 0.2 * rng.standard_normal((m, DV))
    return Segment(USER, keys.copy(), k, v, -np.ones(m, dtype=int))


def evaluate(cache, ep, vocab):
    q = vocab.key_emb[ep.fact_keys[ep.questions]]
    preds = np.array([cache.answer(qi, vocab.val_emb) for qi in q])
    return preds == ep.fact_vals[ep.questions]


def run_episode(ep, vocab, policy, token_budget, bits, setting, rng):
    cache = KVCache(bits=bits)
    turn_ends = set(ep.turn_ends)
    for i, seg in enumerate(ep.segments):
        cache.append(seg)
        if setting == "repeated" and i in turn_ends and len(cache) > token_budget:
            cache.keep(policy(cache, token_budget))
    if setting == "single_shot":
        cache.append(question_segment(ep, vocab, rng))
        if len(cache) > token_budget:
            cache.keep(policy(cache, token_budget))
    return evaluate(cache, ep, vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = EpisodeConfig()
    vocab = Vocab(cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    episodes = [make_episode(cfg, vocab, rng) for _ in range(args.episodes)]
    n_tokens = episodes[0].meta["n_tokens"]
    n_facts = np.mean([len(e.fact_keys) for e in episodes])
    print(f"{args.episodes} episodes, {n_tokens} tokens each, ~{n_facts:.1f} facts each")

    # full-cache reference
    full = np.concatenate([
        run_episode(e, vocab, None, 10**9, 16, "repeated", rng) for e in episodes])
    print(f"Full cache accuracy: {full.mean():.3f}")

    results = {"n_tokens": n_tokens, "n_facts": float(n_facts),
               "full_cache_acc": float(full.mean()), "mem_fracs": MEM_FRACS,
               "settings": {}, "breakdown": {}}
    t0 = time.time()
    for setting in ("single_shot", "repeated"):
        results["settings"][setting] = {}
        for label, pname, bits in RUNS:
            accs = []
            for mf in MEM_FRACS:
                budget = int(mf * n_tokens / bytes_per_token_ratio(bits))
                ok = np.concatenate([
                    run_episode(e, vocab, POLICIES[pname], budget, bits, setting, rng)
                    for e in episodes])
                accs.append(float(ok.mean()))
            results["settings"][setting][label] = accs
            print(f"[{setting:11s}] {label:30s} " +
                  " ".join(f"{a:.2f}" for a in accs))
    print(f"sweep took {time.time() - t0:.0f}s")

    # breakdown for repeated compaction at 6% memory
    mf = 0.06
    for label, pname, bits in RUNS:
        budget = int(mf * n_tokens / bytes_per_token_ratio(bits))
        by_kind, by_ref = {}, {True: [], False: []}
        for e in episodes:
            ok = run_episode(e, vocab, POLICIES[pname], budget, bits, "repeated", rng)
            for o, f in zip(ok, e.questions):
                by_kind.setdefault(TYPE_NAMES[int(e.fact_kind[f])], []).append(o)
                by_ref[bool(e.fact_referenced[f])].append(o)
        results["breakdown"][label] = {
            "by_source": {k: float(np.mean(v)) for k, v in by_kind.items()},
            "referenced_before": float(np.mean(by_ref[True])),
            "never_referenced": float(np.mean(by_ref[False])),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
