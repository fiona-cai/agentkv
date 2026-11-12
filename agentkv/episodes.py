"""Synthetic multi-turn agent episodes.

An episode looks like a real agent transcript, compressed into typed tokens:

    [SYSTEM] -> ( [USER] [ASSISTANT] [TOOL] ) x n_turns -> final questions

Token kinds
-----------
fact      : an addressable "key -> value" memory (e.g. "repo path = /src/app",
            "user wants Python 3.11"). Its key vector lives in the key subspace,
            its value vector in the value subspace.
filler    : ordinary text with weak, random key content.
noise     : tool-output junk (logs, stack traces). Random key content with a
            broad norm distribution, so some junk looks "important".
reference : assistant tokens that re-mention an earlier fact (the agent thinking
            about it). Their *query* points at that fact; they matter for
            attention-based eviction (H2O / SnapKV) because they make the
            referenced fact look important.

At the end, the agent is asked about facts from anywhere in the episode. Some
were referenced along the way, some never were until the question arrives —
exactly the case where "keep what has been attended to so far" breaks.
"""
from dataclasses import dataclass, field
import numpy as np

SYSTEM, USER, ASSISTANT, TOOL = 0, 1, 2, 3
TYPE_NAMES = {SYSTEM: "system", USER: "user", ASSISTANT: "assistant", TOOL: "tool"}

DK = 24  # key/query subspace dimension
DV = 24  # value subspace dimension


@dataclass
class EpisodeConfig:
    n_keys: int = 256          # size of the key vocabulary
    n_values: int = 64         # size of the value vocabulary
    n_turns: int = 12
    system_len: int = 16
    system_facts: int = 4
    user_len: int = 12
    user_facts: int = 2
    assistant_len: int = 16
    assistant_refs: int = 2    # re-mentions of earlier facts per assistant turn
    tool_len: int = 120
    tool_fact_prob: float = 0.6  # chance a tool output contains a useful fact
    n_questions: int = 8


@dataclass
class Segment:
    kind: int
    q: np.ndarray        # (m, DK) queries
    k: np.ndarray        # (m, DK) keys
    v: np.ndarray        # (m, DV) values
    fact_id: np.ndarray  # (m,) fact index or -1


@dataclass
class Episode:
    segments: list                     # list[Segment] in order
    turn_ends: list                    # segment index after which a turn ends
    fact_keys: np.ndarray              # (F,) key id per fact
    fact_vals: np.ndarray              # (F,) value id per fact
    fact_kind: np.ndarray              # (F,) segment type the fact came from
    fact_referenced: np.ndarray        # (F,) bool: re-mentioned before the end
    questions: np.ndarray              # (Q,) fact ids asked at the end
    meta: dict = field(default_factory=dict)


class Vocab:
    """Fixed random embeddings shared by all episodes."""

    def __init__(self, cfg: EpisodeConfig, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.key_emb = _unit(rng.standard_normal((cfg.n_keys, DK)))
        self.val_emb = _unit(rng.standard_normal((cfg.n_values, DV)))


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _junk(rng, m, norm_lo, norm_hi):
    """Random key/value content with a spread of norms."""
    d = _unit(rng.standard_normal((m, DK)))
    n = rng.uniform(norm_lo, norm_hi, size=(m, 1))
    k = d * n
    v = 0.5 * _unit(rng.standard_normal((m, DV)))
    return k, v


def make_episode(cfg: EpisodeConfig, vocab: Vocab, rng) -> Episode:
    segments, turn_ends = [], []
    fact_keys, fact_vals, fact_kind = [], [], []
    used_keys = set()

    def new_fact(kind):
        while True:
            key = int(rng.integers(cfg.n_keys))
            if key not in used_keys:
                used_keys.add(key)
                break
        fact_keys.append(key)
        fact_vals.append(int(rng.integers(cfg.n_values)))
        fact_kind.append(kind)
        return len(fact_keys) - 1

    def text_segment(kind, length, n_facts, junk_norm):
        k, v = _junk(rng, length, *junk_norm)
        q = _junk(rng, length, *junk_norm)[0]
        fid = -np.ones(length, dtype=int)
        slots = rng.choice(length, size=n_facts, replace=False) if n_facts else []
        for s in slots:
            f = new_fact(kind)
            k[s] = vocab.key_emb[fact_keys[f]]
            v[s] = vocab.val_emb[fact_vals[f]]
            q[s] = _junk(rng, 1, *junk_norm)[0][0]
            fid[s] = f
        return Segment(kind, q, k, v, fid)

    referenced = set()
    segments.append(text_segment(SYSTEM, cfg.system_len, cfg.system_facts, (0.1, 0.6)))
    for _ in range(cfg.n_turns):
        segments.append(text_segment(USER, cfg.user_len, cfg.user_facts, (0.1, 0.6)))
        # assistant: mostly filler, a few tokens whose *query* re-mentions a fact
        a = text_segment(ASSISTANT, cfg.assistant_len, 0, (0.1, 0.6))
        n_existing = len(fact_keys)
        if n_existing:
            ref_slots = rng.choice(cfg.assistant_len, size=cfg.assistant_refs, replace=False)
            for s in ref_slots:
                f = int(rng.integers(n_existing))
                a.q[s] = vocab.key_emb[fact_keys[f]]
                referenced.add(f)
        segments.append(a)
        # tool output: long junk with a broad norm spread, maybe one real fact
        n_tool_facts = int(rng.random() < cfg.tool_fact_prob)
        segments.append(text_segment(TOOL, cfg.tool_len, n_tool_facts, (0.1, 1.1)))
        turn_ends.append(len(segments) - 1)

    F = len(fact_keys)
    questions = rng.choice(F, size=min(cfg.n_questions, F), replace=False)
    ref_mask = np.zeros(F, dtype=bool)
    ref_mask[list(referenced)] = True
    return Episode(
        segments=segments,
        turn_ends=turn_ends,
        fact_keys=np.array(fact_keys),
        fact_vals=np.array(fact_vals),
        fact_kind=np.array(fact_kind),
        fact_referenced=ref_mask,
        questions=questions,
        meta={"n_tokens": sum(len(s.fact_id) for s in segments)},
    )
