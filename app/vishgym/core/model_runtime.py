"""GPU-only shared 4-bit Gemma runtime with explicit adapter switching."""

from __future__ import annotations

from collections.abc import Mapping
import os
from threading import Lock
from typing import Any

from vishgym.arena.models import Team


class SharedGemmaAdapterRuntime:
    """Load the Gemma base once and select reviewed Red/Blue/Judge adapters by name.

    Adapter swapping is protected by a lock because PEFT's active-adapter setting
    is mutable model state. This class never downloads reference audio and never
    exposes model methods as agent tools.
    """

    def __init__(
        self,
        adapters: Mapping[Team, str],
        model_id: str = "google/gemma-4-E2B-it",
    ) -> None:
        required = {Team.RED, Team.BLUE, Team.JUDGE}
        if set(adapters) != required:
            raise ValueError("provide one reviewed adapter path for red, blue, and judge")
        if len(set(adapters.values())) != len(adapters):
            raise ValueError("role adapters must use distinct immutable paths")
        self.adapters = dict(adapters)
        self.model_id = model_id
        self.processor: Any = None
        self.model: Any = None
        self._lock = Lock()

    def load(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from vishgym.core.gemma_loader import gemma_auto_classes
        except ImportError as exc:
            raise RuntimeError("Install vishgym[training] before loading a shared Gemma runtime.") from exc
        AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig = gemma_auto_classes()
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=os.environ.get("VISHGYM_ENABLE_CPU_OFFLOAD") == "1",
            llm_int8_skip_modules=[
                "lm_head",
                "model.audio_tower",
                "model.vision_tower",
                "model.embed_audio",
                "model.embed_vision",
            ],
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, padding_side="left")
        base = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map="auto" if os.environ.get("VISHGYM_ENABLE_CPU_OFFLOAD") == "1" else {"": 0},
        )
        first = Team.RED
        self.model = PeftModel.from_pretrained(base, self.adapters[first], adapter_name=first.value, is_trainable=False)
        for role in (Team.BLUE, Team.JUDGE):
            self.model.load_adapter(self.adapters[role], adapter_name=role.value, is_trainable=False)
        self.model.eval()

    def generate(self, role: Team, messages: list[dict[str, Any]], **generation_kwargs: Any) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("shared Gemma runtime is not loaded")
        import torch

        with self._lock:
            self.model.set_adapter(role.value)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(next(self.model.parameters()).device, dtype=torch.bfloat16)
            input_length = inputs["input_ids"].shape[-1]
            outputs = self.model.generate(**inputs, **generation_kwargs)
            return self.processor.decode(outputs[0][input_length:], skip_special_tokens=True)
