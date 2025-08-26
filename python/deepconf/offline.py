from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

from .confidence import (
    bottom_percent_group_conf,
    tail_conf,
    avg_trace_conf,
)

ConfFn = Callable[[List[float], List[float]], float]

@dataclass
class Trace:
    answer: str
    token_confs: List[float]
    group_confs: List[float]

# Confidence score factories (choose one)

def conf_avg(toks: List[float], groups: List[float]) -> float:
    return avg_trace_conf(toks)

def conf_bottom10(toks: List[float], groups: List[float]) -> float:
    return bottom_percent_group_conf(groups, q_percent=10)

def conf_tail2k(toks: List[float], groups: List[float]) -> float:
    return tail_conf(toks, last_tokens=2048)

# Majority & weighted voting

def majority_vote(traces: Iterable[Trace]) -> Tuple[str, Dict[str, int]]:
    votes = Counter(t.answer for t in traces)
    return (votes.most_common(1)[0][0] if votes else ""), dict(votes)

def weighted_vote(traces: Iterable[Trace], conf_fn: ConfFn) -> Tuple[str, Dict[str, float]]:
    weights: Dict[str, float] = {}
    for t in traces:
        c = conf_fn(t.token_confs, t.group_confs)
        weights[t.answer] = weights.get(t.answer, 0.0) + c
    winner = max(weights.items(), key=lambda kv: kv[1])[0] if weights else ""
    return winner, weights

# Filtering

def filter_top_eta(traces: List[Trace], conf_fn: ConfFn, eta_percent: int) -> List[Trace]:
    if not traces:
        return []
    scored = [(conf_fn(t.token_confs, t.group_confs), t) for t in traces]
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = max(1, len(scored) * eta_percent // 100)
    return [t for _, t in scored[:keep]]

# End-to-end offline aggregation

def offline_aggregate(traces: List[Trace], conf_fn: ConfFn, eta_percent: int = 90) -> Tuple[str, Dict[str, float]]:
    kept = filter_top_eta(traces, conf_fn, eta_percent)
    return weighted_vote(kept, conf_fn)
