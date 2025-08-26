from __future__ import annotations
import re
from typing import Any, Dict, Optional

DEFAULTS = {
    "enabled": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_logprobs": 20,
    # Your request:
    "group_window_target": 100000,
    # DeepConf ensemble / online settings
    "eta_percent": 10,              # keep top-η% traces (aggressive; safer is 90)
    "consensus_threshold": 0.95,    # stop when modal answer weight ≥ τ
    "warmup_traces": 16,
    "max_budget": 512,
    # Safety rails for huge windows
    "min_effective_window": 512,
    "absolute_window_cap": 131072,  # 128k cap
    # Model-specific nudges (regex keys)
    "model_overrides": {
        r"gpt[-_]?oss[-_]?120b": {
            "eta_percent": 10,
            "consensus_threshold": 0.95,
            "group_window_target": 100000,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_logprobs": 20,
        }
    },
}

def merge_deep(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_deep(out[k], v)
        else:
            out[k] = v
    return out

def resolve_for_model(model_name: str, user_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = merge_deep(DEFAULTS, user_cfg or {})
    for pat, ov in cfg.get("model_overrides", {}).items():
        if re.search(pat, model_name or "", re.IGNORECASE):
            cfg = merge_deep(cfg, ov)
    return cfg

def effective_group_window(target: int,
                           provider_ctx_limit: Optional[int],
                           tokens_seen: int,
                           min_effective: int,
                           absolute_cap: int) -> int:
    # Cap by provider context (if known) and our absolute bound.
    # Use half the provider ctx to avoid colliding with KV cache / other buffers.
    half_ctx = (provider_ctx_limit or absolute_cap) // 2
    ctx_cap = max(min(absolute_cap, half_ctx), min_effective)
    hard_cap = max(min_effective, min(target, ctx_cap))
    # Don’t exceed tokens we’ve actually seen (startup: grow window gradually)
    return max(min_effective, min(hard_cap, max(min_effective, tokens_seen)))
