"""AgentKV as a KVPress "press" for real Hugging Face models.

STATUS: UNTESTED SCAFFOLD. Written against NVIDIA/kvpress's ScorerPress
interface but not run here (no GPU / torch in the prototype environment).
Check the current kvpress API before relying on it:
https://github.com/NVIDIA/kvpress

Idea: take any query-agnostic base scorer that KVPress already ships, normalise
its scores per head, and add a bonus based on what *kind* of agent text each
token belongs to (system prompt / user / assistant / tool output). In the
project, the hand-set TYPE_PRIOR gets replaced by a small learned scorer.

Note: the NumPy sim uses "key norm" as its content signal because of how the
sim is built. Don't copy that to real models: in real LLMs, low-norm keys often
get *more* attention. Use a proper base press instead (default below).

Usage sketch:
    from kvpress import ExpectedAttentionPress
    from agentkv_press import AgentKVPress, token_types_from_chat

    press = AgentKVPress(base=ExpectedAttentionPress(), compression_ratio=0.8)
    press.token_types = token_types_from_chat(tokenizer, messages)  # (seq_len,)
    with press(model):
        model(input_ids)  # prefill: cache is compressed here
"""
from dataclasses import dataclass, field

import torch
from kvpress import ScorerPress

SYSTEM, USER, ASSISTANT, TOOL = 0, 1, 2, 3
TYPE_PRIOR = {SYSTEM: 2.0, USER: 1.5, ASSISTANT: 0.5, TOOL: 0.0}


@dataclass
class AgentKVPress(ScorerPress):
    base: ScorerPress = None
    compression_ratio: float = 0.0
    type_weight: float = 1.0
    token_types: torch.Tensor = field(default=None, repr=False)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        base = self.base.score(module, hidden_states, keys, values, attentions, kwargs)
        # z-score per (batch, head) so the type bonus has a consistent scale
        mu = base.mean(dim=-1, keepdim=True)
        sd = base.std(dim=-1, keepdim=True).clamp_min(1e-6)
        scores = (base - mu) / sd
        if self.token_types is not None:
            seq_len = keys.shape[-2]
            types = self.token_types[-seq_len:].to(keys.device)
            prior = torch.tensor([TYPE_PRIOR[int(t)] for t in types.tolist()],
                                 device=keys.device, dtype=scores.dtype)
            scores = scores + self.type_weight * prior.view(1, 1, -1)
        return scores


def token_types_from_chat(tokenizer, messages):
    """Label every token of a chat-templated conversation with its role.

    messages: [{"role": "system"|"user"|"assistant"|"tool", "content": str}, ...]
    Tokenises message-by-message so labels line up with the template output.
    Verify alignment for your tokenizer: some templates merge tokens across turns.
    """
    role_to_type = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT, "tool": TOOL}
    types, prev_len = [], 0
    for i in range(len(messages)):
        ids = tokenizer.apply_chat_template(messages[: i + 1], tokenize=True)
        types += [role_to_type[messages[i]["role"]]] * (len(ids) - prev_len)
        prev_len = len(ids)
    return torch.tensor(types)
