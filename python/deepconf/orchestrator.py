from __future__ import annotations
from typing import Any, Dict, Tuple

from .settings import resolve_for_model
from .online import OnlineDeepConf
from .provider_adapter import ProviderAdapter

class DeepConfOrchestrator:
    def __init__(self, model_wrapper: Any, deepconf_cfg: Dict, model_name: str):
        # Prefer a raw client if your wrapper exposes it; otherwise, it can be the wrapper itself if it proxies .chat.completions
        raw_client = getattr(model_wrapper, "raw_client", None) or getattr(model_wrapper, "_client", None) or model_wrapper
        self.adapter = ProviderAdapter(raw_client, model_name, deepconf_cfg)
        self.cfg = resolve_for_model(model_name, deepconf_cfg)

    async def run(self, messages: list[dict]) -> Tuple[str, Dict[str, float]]:
        dc = OnlineDeepConf(self.adapter, self.adapter.model_name, self.cfg)
        return await dc.solve(messages)
