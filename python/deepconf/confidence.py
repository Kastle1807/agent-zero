from __future__ import annotations
from collections import deque
from statistics import mean
from typing import Deque, List, Optional

from .settings import effective_group_window

class MovingConfidence:
    """
    Maintains a moving confidence window; used for Lowest Group Confidence (LGC)
    by treating the window-average as the group confidence at each step.
    """
    def __init__(self, target_window: int, min_effective: int,
                 absolute_cap: int, provider_ctx_limit: Optional[int] = None):
        self.target_window = target_window
        self.min_effective = min_effective
        self.absolute_cap = absolute_cap
        self.provider_ctx_limit = provider_ctx_limit
        self.tokens_seen = 0
        self.queue: Deque[float] = deque()
        self.sum_vals: float = 0.0

    def push(self, token_conf: float):
        self.tokens_seen += 1
        eff = effective_group_window(
            self.target_window, self.provider_ctx_limit, self.tokens_seen,
            self.min_effective, self.absolute_cap,
        )
        # Shrink if needed
        while len(self.queue) > eff:
            self.sum_vals -= self.queue.popleft()
        # Grow with backfill to avoid bias
        while len(self.queue) < eff:
            self.queue.append(token_conf)
            self.sum_vals += token_conf
        # Slide
        if self.queue:
            self.sum_vals -= self.queue.pop()
        self.queue.append(token_conf)
        self.sum_vals += token_conf

    def group_conf(self) -> float:
        if not self.queue:
            return float("inf")
        return self.sum_vals / len(self.queue)

# --- Trace-level aggregations ---

def bottom_percent_group_conf(group_conf_list: List[float], q_percent: int = 10) -> float:
    if not group_conf_list:
        return float("inf")
    k = max(1, len(group_conf_list) * q_percent // 100)
    lows = sorted(group_conf_list)[:k]
    return mean(lows)

def tail_conf(token_conf_list: List[float], last_tokens: int = 2048) -> float:
    if not token_conf_list:
        return float("inf")
    toks = token_conf_list[-min(last_tokens, len(token_conf_list)) :]
    return mean(toks)


def avg_trace_conf(token_conf_list: List[float]) -> float:
    return mean(token_conf_list) if token_conf_list else float("inf")
