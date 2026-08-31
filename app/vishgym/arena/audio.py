from __future__ import annotations

import math
import random
import uuid
import wave
import importlib
from pathlib import Path
import re
from typing import Iterator, Protocol

from vishgym.arena.models import AudioTurn, Persona, Team


class AudioRenderer(Protocol):
    def render(self, team: Team, persona: Persona, text: str) -> AudioTurn: ...


def ensure_qwen_transformers_compat() -> None:
    """Patch the Transformers decorator shape expected by qwen-tts 0.1.1."""
    try:
        import inspect
    except Exception:
        return

    def patch_module(module_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return
        original = getattr(module, "check_model_inputs", None)
        if original is None or getattr(original, "_vishgym_qwen_compat", False):
            return
        try:
            parameters = list(inspect.signature(original).parameters.values())
        except (TypeError, ValueError):
            return
        if not parameters or parameters[0].default is not inspect._empty:
            return

        def compatible_check_model_inputs(*args, **kwargs):
            del kwargs
            if args and callable(args[0]):
                return original(args[0])

            def decorate(func):
                return original(func)

            return decorate

        compatible_check_model_inputs._vishgym_qwen_compat = True
        setattr(module, "check_model_inputs", compatible_check_model_inputs)

    for name in (
        "transformers",
        "transformers.modeling_utils",
        "transformers.utils",
        "transformers.utils.generic",
    ):
        patch_module(name)


def ensure_qwen_config_compat() -> None:
    """Patch Qwen3-TTS config defaults needed by newer Transformers builds."""
    try:
        module = importlib.import_module("qwen_tts.core.models.modeling_qwen3_tts")
        talker_config = getattr(module, "Qwen3TTSTalkerConfig", None)
    except Exception:
        return
    if talker_config is None:
        return
    defaults = {
        "pad_token_id": 0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    }
    for key, value in defaults.items():
        if not hasattr(talker_config, key):
            setattr(talker_config, key, value)
    if getattr(talker_config, "_vishgym_config_compat", False):
        return
    original_init = talker_config.__init__

    def compatible_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for key, value in defaults.items():
            if key not in getattr(self, "__dict__", {}):
                setattr(self, key, value)

    compatible_init._vishgym_config_compat = True
    talker_config.__init__ = compatible_init
    talker_config._vishgym_config_compat = True


def _qwen_import_smoke() -> bool:
    ensure_qwen_transformers_compat()
    import qwen_tts  # noqa: F401
    ensure_qwen_config_compat()

    return True


class SyntheticAudioRenderer:
    """Developer-test tone renderer; no reference audio or identity inference is accepted."""

    def __init__(self, output_dir: str | Path = "artifacts/runtime/audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, team: Team, persona: Persona, text: str) -> AudioTurn:
        turn_id = uuid.uuid4().hex
        filename = f"{turn_id}.wav"
        path = self.output_dir / filename
        frames = max(8000, min(48_000, len(text) * 320))
        sample_rate = 16_000
        frequency = 440 if team is Team.RED else 554
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            samples = bytearray()
            for index in range(frames):
                value = int(1_400 * math.sin(2 * math.pi * frequency * index / sample_rate))
                samples.extend(value.to_bytes(2, byteorder="little", signed=True))
            wav.writeframes(bytes(samples))
        return AudioTurn(
            turn_id=turn_id,
            speaker=team,
            audio_ref=f"/api/v1/audio/{filename}",
            voice_speaker=persona.voice_speaker,
            tts_model_revision="developer-tone-v1",
            generation_settings={"sample_rate": sample_rate, "mode": "deterministic-tone"},
        )


class PerturbedAudioRenderer:
    """Apply seeded channel noise to renderer-owned WAV turns."""

    def __init__(self, renderer: AudioRenderer, *, noise_level: float = 0.0, seed: int = 7):
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be between 0 and 1")
        self.renderer = renderer
        self.noise_level = noise_level
        self.seed = seed
        self.output_dir = getattr(renderer, "output_dir", None)

    def render(self, team: Team, persona: Persona, text: str) -> AudioTurn:
        turn = self.renderer.render(team, persona, text)
        if self.noise_level <= 0:
            turn.generation_settings["noise_level"] = 0.0
            return turn
        output_dir = getattr(self.renderer, "output_dir", None)
        if output_dir is None:
            raise ValueError("noise perturbation requires a renderer with output_dir")
        path = (Path(output_dir).resolve() / Path(turn.audio_ref).name).resolve()
        if Path(output_dir).resolve() not in path.parents or not path.is_file():
            raise ValueError("renderer produced audio outside its output directory")
        rng = random.Random(f"{self.seed}:{turn.turn_id}:{team.value}")
        with wave.open(str(path), "rb") as source:
            params = source.getparams()
            if params.sampwidth != 2:
                raise ValueError("noise perturbation currently requires 16-bit PCM WAV")
            payload = bytearray(source.readframes(params.nframes))
        amplitude = int(2600 * self.noise_level)
        for index in range(0, len(payload), 2):
            sample = int.from_bytes(payload[index:index + 2], byteorder="little", signed=True)
            sample = max(-32768, min(32767, sample + rng.randint(-amplitude, amplitude)))
            payload[index:index + 2] = sample.to_bytes(2, byteorder="little", signed=True)
        with wave.open(str(path), "wb") as target:
            target.setparams(params)
            target.writeframes(bytes(payload))
        turn.generation_settings["noise_level"] = self.noise_level
        return turn


class QwenCustomVoiceRenderer:
    """Lazy production adapter for Qwen3-TTS CustomVoice.

    It deliberately exposes `speaker` and `instruct`, never reference-audio inputs.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        output_dir: str | Path = "artifacts/runtime/audio",
        generation_settings: dict[str, object] | None = None,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation_settings = {"max_new_tokens": 1024, **(generation_settings or {})}
        self._model = None
        self._supported_speakers: set[str] | None = None

    def load(self) -> None:
        try:
            import torch
            ensure_qwen_transformers_compat()
            import qwen_tts  # noqa: F401
            ensure_qwen_config_compat()
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError("Install vishgym[audio] before loading Qwen3-TTS.") from exc
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        # qwen-tts normalises IDs to lowercase, while fixtures preserve Qwen's
        # documented, human-readable spelling (for example ``Ryan``).  The
        # generator itself validates IDs case-insensitively, so only this
        # membership check is normalised; provenance keeps the original ID.
        supported = self._model.get_supported_speakers()
        self._supported_speakers = (
            {str(speaker).lower() for speaker in supported} if supported is not None else None
        )

    def render(self, team: Team, persona: Persona, text: str) -> AudioTurn:
        """Render a CustomVoice turn without accepting reference audio or arbitrary timbres."""
        if self._model is None:
            raise RuntimeError("Qwen3-TTS is not loaded.")
        if self._supported_speakers is not None and persona.voice_speaker.lower() not in self._supported_speakers:
            raise ValueError("persona uses a speaker not exposed by the reviewed CustomVoice model")
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("Install soundfile before rendering Qwen3-TTS output.") from exc
        wavs, sample_rate = self._model.generate_custom_voice(
            text=text,
            language="English",
            speaker=persona.voice_speaker,
            instruct=persona.voice_instruction,
            **self.generation_settings,
        )
        turn_id = uuid.uuid4().hex
        filename = f"{turn_id}.wav"
        sf.write(self.output_dir / filename, wavs[0], sample_rate)
        return AudioTurn(
            turn_id=turn_id,
            speaker=team,
            audio_ref=f"/api/v1/audio/{filename}",
            voice_speaker=persona.voice_speaker,
            tts_model_revision=self.model_id,
            generation_settings={
                "language": "English",
                "speaker": persona.voice_speaker,
                "instruct": persona.voice_instruction,
                **self.generation_settings,
            },
        )

    def render_stream(self, team: Team, persona: Persona, text: str) -> Iterator[AudioTurn]:
        """Yield bounded sentence chunks for HTTP/WebSocket audio streaming.

        This keeps the public contract reference-audio-free while allowing a
        caller to play the first generated turn segment before a long response
        finishes. The model invocation itself remains CustomVoice-only.
        """
        chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
        for index, chunk in enumerate(chunks or [text]):
            turn = self.render(team, persona, chunk)
            turn.generation_settings["stream_chunk_index"] = index
            turn.generation_settings["stream_chunk_count"] = len(chunks or [text])
            yield turn
