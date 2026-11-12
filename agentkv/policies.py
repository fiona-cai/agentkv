"""KV-cache eviction policies.

Each policy maps (cache, token_budget) -> indices of entries to keep.
Baselines are simplified versions of published methods:

  recent      : sliding window (keep the last B tokens)
  streaming   : StreamingLLM — attention sinks (first 4) + recent window
  h2o         : H2O — heavy hitters by accumulated attention + recent window
  snapkv      : SnapKV — score by attention from the last few queries
  agent_type  : AgentKV, type prior only (system > user > assistant > tool)
  agent_addr  : AgentKV, addressability only (query-agnostic key norm)
  agentkv     : AgentKV hybrid — type prior + addressability + recency

The AgentKV weights below are hand-set. In the real project they'd be learned
by a small scorer trained on agent trajectories.
"""
import numpy as np
from .cache import GAIN, softmax
from .episodes import SYSTEM, USER, ASSISTANT, TOOL


def _recent_idx(cache, r):
    return np.arange(max(0, len(cache) - r), len(cache))


def _top_with_recent(cache, budget, scores, recent_frac):
    r = int(budget * recent_frac)
    recent = _recent_idx(cache, r)
    rest = np.setdiff1d(np.arange(len(cache)), recent)
    n_pick = budget - len(recent)
    picked = rest[np.argsort(-scores[rest], kind="stable")[:n_pick]]
    return np.concatenate([recent, picked])


def recent(cache, budget):
    return _recent_idx(cache, budget)


def streaming(cache, budget, sinks=4):
    return np.concatenate([np.arange(sinks), _recent_idx(cache, budget - sinks)])


def h2o(cache, budget):
    return _top_with_recent(cache, budget, cache.acc_attn, recent_frac=0.25)


def snapkv(cache, budget):
    obs = softmax(GAIN * cache.recent_q @ cache.k.T).sum(axis=0)
    return _top_with_recent(cache, budget, obs, recent_frac=0.125)


TYPE_PRIOR = {SYSTEM: 1.0, USER: 0.8, ASSISTANT: 0.3, TOOL: 0.0}


def _type_score(cache):
    return np.vectorize(TYPE_PRIOR.get)(cache.kind).astype(float)


def _addr_score(cache):
    return np.linalg.norm(cache.k, axis=1)


def _recency_score(cache):
    return cache.pos / max(1, cache.t)


def agent_type(cache, budget):
    s = _type_score(cache) + 0.01 * _recency_score(cache)
    return _top_with_recent(cache, budget, s, recent_frac=0.125)


def agent_addr(cache, budget):
    s = _addr_score(cache) + 0.01 * _recency_score(cache)
    return _top_with_recent(cache, budget, s, recent_frac=0.125)


def agentkv(cache, budget, w_type=1.0, w_addr=1.5, w_rec=0.05):
    s = (w_type * _type_score(cache)
         + w_addr * _addr_score(cache)
         + w_rec * _recency_score(cache))
    return _top_with_recent(cache, budget, s, recent_frac=0.125)


POLICIES = {
    "recent": recent,
    "streaming": streaming,
    "h2o": h2o,
    "snapkv": snapkv,
    "agent_type": agent_type,
    "agent_addr": agent_addr,
    "agentkv": agentkv,
}
