from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .confidence import MovingConfidence
from .offline import Trace, offline_aggregate, conf_bottom10, conf_tail2k, conf_avg

@dataclass
class OnlineSettings:
    eta_percent: int  # top-η% kept during aggregation
    consensus_threshold: float  # τ for stop on consensus
    warmup_traces: int  # N_init
    max_budget: int  # K
    group_window_target: int
    min_effective_window: int
    absolute_window_cap: int

ConfSelector = Callable[[List[float], List[float]], float]

class OnlineDeepConf:
    def __init__(self, provider_adapter, model_name: str, cfg: Dict):
        self.adapter = provider_adapter
        self.model_name = model_name
        self.cfg = cfg
        self.settings = OnlineSettings(
            eta_percent=cfg["eta_percent"],
            consensus_threshold=cfg["consensus_threshold"],
            warmup_traces=cfg["warmup_traces"],
            max_budget=cfg["max_budget"],
            group_window_target=cfg["group_window_target"],
            min_effective_window=cfg["min_effective_window"],
            absolute_window_cap=cfg["absolute_window_cap"],
        )
        # default confidence rule for online thresholding
        self.conf_fn: ConfSelector = conf_bottom10

    # --- Helpers ---
    def _token_conf_from_logprobs(self, top_candidates: List[Tuple[str, float]]) -> float:
        # Use negative mean logprob of the top-k candidate list as a proxy for confidence
        if not top_candidates:
            return 0.0
        return -sum(lp for _, lp in top_candidates) / len(top_candidates)

    async def _gen_one_trace(self, messages: list[dict], stop_threshold: Optional[float]) -> Trace:
        mv = MovingConfidence(
            target_window=self.settings.group_window_target,
            min_effective=self.settings.min_effective_window,
            absolute_cap=self.settings.absolute_window_cap,
            provider_ctx_limit=getattr(self.adapter, "provider_ctx_limit", None),
        )
        token_confs: List[float] = []
        group_confs: List[float] = []
        answer_text_chunks: List[str] = []

        async for chunk, lp in self.adapter.stream_chat(messages):
            if not chunk:
                continue
            answer_text_chunks.append(chunk)

            # Expect provider to give a structure with candidates (token, logprob)
            topk = []
            if lp and hasattr(lp, "content") and lp.content:
                # Some providers put per-token data in lp.content[-1]
                entry = lp.content[-1]
                # try common shape: entry.top_logprobs: List[{token, logprob}]
                topk = [ (d.get("token"), d.get("logprob", 0.0)) for d in entry.get("top_logprobs", []) ]
            elif isinstance(lp, dict) and "top_logprobs" in lp:
                topk = [ (d.get("token"), d.get("logprob", 0.0)) for d in lp.get("top_logprobs", []) ]

            c_tok = self._token_conf_from_logprobs(topk)
            token_confs.append(c_tok)
            mv.push(c_tok)
            group_confs.append(mv.group_conf())

            # Early stop if online threshold supplied and confidence drops below it
            if stop_threshold is not None and mv.group_conf() < stop_threshold and len(token_confs) >= self.settings.min_effective_window:
                break

        # Extract final answer string from chunks (your post-processing can be more sophisticated)
        answer = "".join(answer_text_chunks)
        return Trace(answer=answer, token_confs=token_confs, group_confs=group_confs)

    # --- Public entrypoint ---
    async def solve(self, messages: list[dict]) -> Tuple[str, Dict[str, float]]:
        traces: List[Trace] = []

        # 1) Warmup: run N_init full traces to set threshold s over top-η% traces
        for _ in range(self.settings.warmup_traces):
            t = await self._gen_one_trace(messages, stop_threshold=None)
            traces.append(t)

        # compute s: percentile boundary to keep top-η% by our selected confidence
        # we’ll use Bottom-10% group confidence as the trace-level score during online
        scores = [ self.conf_fn(t.token_confs, t.group_confs) for t in traces ]
        if not scores:
            # No signal — fallback to single majority vote
            return offline_aggregate(traces, self.conf_fn, self.settings.eta_percent)

        # threshold s is the minimum score among the top-η%
        keep = max(1, len(scores) * self.settings.eta_percent // 100)
        s = sorted(scores, reverse=True)[:keep][-1]
        # expose to adapter for optional server-side gating
        setattr(self.adapter, "current_threshold", float(s))

        # 2) Adaptive sampling until consensus or budget
        while len(traces) < self.settings.max_budget:
            # Check consensus on current set
            winner, weights = offline_aggregate(traces, self.conf_fn, self.settings.eta_percent)
            total = sum(weights.values()) if weights else 0.0
            conf = (weights.get(winner, 0.0) / total) if total > 0 else 0.0
            if conf >= self.settings.consensus_threshold:
                return winner, weights

            # Otherwise, generate another trace with online early-stop at s
            t = await self._gen_one_trace(messages, stop_threshold=s)
            traces.append(t)

        # 3) Budget reached; return aggregate
        return offline_aggregate(traces, self.conf_fn, self.settings.eta_percent)
