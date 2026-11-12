"""A KV cache with a single planted retrieval attention head.

The "model" is one attention head whose query/key projections read the key
subspace and whose value projection reads the value subspace. It's the minimal
mechanism that makes long-context recall depend on which cache entries survive
compression. Real LLMs have many such heads; this keeps the sim runnable on a
laptop CPU with NumPy only.
"""
import numpy as np

GAIN = 12.0  # attention temperature (sharpness of retrieval)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fake_quant(x, bits):
    """Per-token asymmetric min-max quantization (KIVI-style for values)."""
    if bits >= 16:
        return x
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    levels = 2 ** bits - 1
    scale = np.where(hi > lo, (hi - lo) / levels, 1.0)
    return np.round((x - lo) / scale) * scale + lo


def bytes_per_token_ratio(bits, d=24):
    """Memory per cached token relative to fp16, incl. fp16 scale+zero per vector."""
    if bits >= 16:
        return 1.0
    return (d * bits + 2 * 16) / (d * 16)


class KVCache:
    def __init__(self, bits=16):
        self.bits = bits
        self.k = np.zeros((0, 24))
        self.v = np.zeros((0, 24))
        self.kind = np.zeros(0, dtype=int)
        self.pos = np.zeros(0, dtype=int)
        self.fact_id = np.zeros(0, dtype=int)
        self.acc_attn = np.zeros(0)      # H2O: total attention received so far
        self.recent_q = np.zeros((0, 24))  # SnapKV: last few queries seen
        self.t = 0

    def __len__(self):
        return len(self.pos)

    def append(self, seg, window=8):
        """Process a segment causally: attend, accumulate scores, then cache it."""
        m = len(seg.fact_id)
        k_new = fake_quant(seg.k, self.bits)
        v_new = fake_quant(seg.v, self.bits)
        K = np.concatenate([self.k, k_new])
        n_old = len(self)
        scores = GAIN * seg.q @ K.T                      # (m, n_old + m)
        causal = np.tril(np.ones((m, m), dtype=bool))
        mask = np.concatenate([np.ones((m, n_old), dtype=bool), causal], axis=1)
        scores = np.where(mask, scores, -np.inf)
        attn = softmax(scores)
        received = attn.sum(axis=0)

        self.k = K
        self.v = np.concatenate([self.v, v_new])
        self.kind = np.concatenate([self.kind, np.full(m, seg.kind)])
        self.pos = np.concatenate([self.pos, np.arange(self.t, self.t + m)])
        self.fact_id = np.concatenate([self.fact_id, seg.fact_id])
        self.acc_attn = np.concatenate([self.acc_attn, np.zeros(m)]) + received
        self.recent_q = np.concatenate([self.recent_q, seg.q])[-window:]
        self.t += m

    def keep(self, idx):
        idx = np.sort(idx)
        for name in ("k", "v", "kind", "pos", "fact_id", "acc_attn"):
            setattr(self, name, getattr(self, name)[idx])

    def answer(self, query, val_emb):
        """Retrieve with one query vector; return predicted value id."""
        if len(self) == 0:
            return -1
        attn = softmax(GAIN * self.k @ query)
        out = attn @ self.v
        return int(np.argmax(val_emb @ out))
