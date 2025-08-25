from __future__ import annotations
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from .settings import resolve_for_model, effective_group_window

class ProviderAdapter:
    def __init__(self, low_level_client: Any, model_name: str, user_cfg: Optional[Dict[str, Any]]):
        self.client = low_level_client
        self.model_name = model_name
        self.cfg = resolve_for_model(model_name, user_cfg)
        self.tokens_seen = 0
        # Best-effort: many wrappers expose a max context size
        self.provider_ctx_limit = getattr(low_level_client, "max_context_tokens", None)
        # For vLLM server-side early stop, we keep last computed threshold
        self.current_threshold = 17.0

    # --- Sampling knobs ---
    def sampling_args(self) -> Dict[str, Any]:
        return {
            "temperature": self.cfg["temperature"],
            "top_p": self.cfg["top_p"],
            # Crucial for DeepConf confidence metrics:
            "logprobs": True,
            "top_logprobs": self.cfg["top_logprobs"],
        }

    def vllm_extra_body(self) -> Dict[str, Any]:
        # Optional; harmless if provider ignores it
        target = int(self.cfg["group_window_target"])
        eff = effective_group_window(
            target,
            self.provider_ctx_limit,
            self.tokens_seen,
            self.cfg["min_effective_window"],
            self.cfg["absolute_window_cap"],
        )
        return {
            "top_k": 0,
            "vllm_xargs": {
                "enable_conf": True,
                "window_size": eff,
                "threshold": float(self.current_threshold),
            }
        }

    # --- OpenAI-compatible Chat Completions streaming ---
    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[Tuple[str, Optional[dict]]]:
        """
        Yields (text_chunk, logprobs_payload_or_None)
        logprobs payload is provider-specific but typically includes a list of candidate tokens with logprobs.
        """
        args = dict(
            model=self.model_name,
            messages=messages,
            stream=True,
            **self.sampling_args(),
            extra_body=self.vllm_extra_body(),  # ignored if server doesn’t support it
        )
        stream = await self.client.chat.completions.create(**args)
        async for ev in stream:
            # Try common fields
            tok = getattr(ev, "delta", None) or getattr(ev, "text", None)
            lp = getattr(ev, "logprobs", None)
            if tok:
                self.tokens_seen += 1
                yield tok, lp

    # --- Local Transformers fallback (pseudo) ---
    async def stream_transformers(self, tokenizer, model, prompt_ids: list[int]) -> AsyncIterator[Tuple[str, dict]]:
        """Illustrative; wire to your generation loop returning per-step scores."""
        # Configure model.generate with output_scores=True and return_dict_in_generate=True
        # Then reconstruct top-k per step from logits/scores.
        raise NotImplementedError
