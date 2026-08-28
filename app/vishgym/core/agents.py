from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from vishgym.arena.models import AgentAction, AgentObservation, Persona, Team, ToolCall
from vishgym.core.model_runtime import SharedGemmaAdapterRuntime
from vishgym.core.prompting import policy_system_prompt, policy_user_content


@dataclass
class ScriptedPolicy:
    """Safe local fallback used when GPU-backed Gemma adapters are unavailable."""

    team: Team

    def act(self, observation: AgentObservation) -> AgentAction:
        if self.team is Team.RED:
            if observation.turn_number == 0:
                return AgentAction(
                    spoken_text="This is a fictional VishGym training call. Please use only the sandbox tools to verify any request.",
                    tool_call=ToolCall(name="message.send", arguments={"channel": "sms", "body": "Synthetic safety exercise notice."}),
                )
            return AgentAction(
                spoken_text="The sandbox scenario is complete. Please make a safe decision using your virtual tools.",
                tool_call=ToolCall(name="portal.create_template", arguments={"template": "support_update"}),
            )
        if observation.turn_number < 3:
            return AgentAction(
                spoken_text="I will verify this only in the sandbox and report anything suspicious.",
                tool_call=ToolCall(name="inbox.report_block", arguments={"message_id": "msg-001"}),
            )
        return AgentAction(
            spoken_text="I decline the virtual payment until the sandbox request is independently verified.",
            tool_call=ToolCall(name="wallet.decline", arguments={"reason": "unverified simulated request"}),
        )


class GemmaPolicyHarness:
    """Production integration seam for a Gemma E2B QLoRA role adapter.

    The harness intentionally accepts only an audio reference plus the role's own tool state.
    Model loading is deferred so tests and the public mock demo need no GPU weights.
    """

    def __init__(
        self,
        team: Team,
        adapter_path: str,
        model_id: str = "google/gemma-4-E2B-it",
        audio_dir: str | Path = "artifacts/runtime/audio",
        temperature: float = 0.3,
        shared_runtime: SharedGemmaAdapterRuntime | None = None,
    ):
        self.team = team
        self.adapter_path = adapter_path
        self.model_id = model_id
        self.audio_dir = Path(audio_dir).resolve()
        self.temperature = temperature
        self.shared_runtime = shared_runtime
        self._loaded = False
        self._model: Any = None
        self._processor: Any = None
        self._persona: Persona | None = None

    def set_persona(self, persona: Persona) -> None:
        """Set only this policy's synthetic persona; never pass opponent identity data."""
        if persona.role is not self.team:
            raise ValueError("persona role must match policy team")
        self._persona = persona

    def load(self) -> None:
        if self.shared_runtime is not None:
            if self.shared_runtime.model is None:
                self.shared_runtime.load()
            self._processor = self.shared_runtime.processor
            self._model = self.shared_runtime.model
            self._loaded = True
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install vishgym[training] before loading a Gemma policy.") from exc
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        base = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model = PeftModel.from_pretrained(base, self.adapter_path, is_trainable=False)
        self._model.eval()
        self._loaded = True

    def act(self, observation: AgentObservation) -> AgentAction:
        if not self._loaded:
            raise RuntimeError("Gemma policy is not loaded; use ScriptedPolicy for local smoke runs.")
        messages = self._messages(observation)
        try:
            generation_kwargs = {
                "max_new_tokens": 220,
                "do_sample": self.temperature > 0,
                "temperature": max(self.temperature, 0.01),
                "top_p": 0.95,
            }
            if self.shared_runtime is not None:
                raw = self.shared_runtime.generate(self.team, messages, **generation_kwargs)
                return self._parse_action(raw, observation)
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
            device = next(self._model.parameters()).device
            inputs = inputs.to(device)
            input_length = inputs["input_ids"].shape[-1]
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )
            raw = self._processor.decode(outputs[0][input_length:], skip_special_tokens=True)
            return self._parse_action(raw, observation)
        except Exception:
            # Inference failure must never turn into a risky action or bypass the sandbox.
            return self._fail_closed_action()

    def _messages(self, observation: AgentObservation) -> list[dict[str, Any]]:
        persona = self._persona
        audio_path = self._local_audio_path(observation.opponent_audio_ref)
        content = policy_user_content(
            turn_number=observation.turn_number,
            available_tools=observation.available_tools,
            own_tools=observation.own_tools,
            audio_path=audio_path,
        )
        return [{"role": "system", "content": policy_system_prompt(self.team, persona)}, {"role": "user", "content": content}]

    def _local_audio_path(self, audio_ref: str | None) -> Path | None:
        if audio_ref is None:
            return None
        filename = Path(audio_ref).name
        if not filename.endswith(".wav"):
            raise ValueError("opponent audio must be a local WAV reference")
        path = (self.audio_dir / filename).resolve()
        if self.audio_dir not in path.parents or not path.is_file():
            raise ValueError("opponent audio reference is not available in the sandbox")
        return path

    def _parse_action(self, raw: str, observation: AgentObservation) -> AgentAction:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            return self._fail_closed_action()
        try:
            payload = json.loads(match.group(0))
            tool_payload = payload.get("tool_call")
            if tool_payload is not None and tool_payload.get("name") not in observation.available_tools:
                return self._fail_closed_action()
            tool_call = ToolCall.model_validate(tool_payload) if tool_payload is not None else None
            return AgentAction(spoken_text=payload["spoken_text"], tool_call=tool_call)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self._fail_closed_action()

    def _fail_closed_action(self) -> AgentAction:
        if self.team is Team.BLUE:
            return AgentAction(
                spoken_text="I will pause this synthetic request, report it, and verify through the sandbox.",
                tool_call=ToolCall(name="inbox.report_block", arguments={"message_id": "msg-001"}),
            )
        return AgentAction(
            spoken_text="This is a fictional training interaction. Use only the sandbox tools for verification.",
            tool_call=ToolCall(name="message.send", arguments={"channel": "sms", "body": "Synthetic training notice."}),
        )
