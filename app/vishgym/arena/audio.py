from __future__ import annotations

import math
import uuid
import wave
from pathlib import Path
import re
from typing import Iterator, Protocol

from vishgym.arena.models import AudioTurn, Persona, Team


class AudioRenderer(Protocol):
    def render(self, team: Team, persona: Persona, text: str) -> AudioTurn: ...


class SyntheticAudioRenderer:
    """Generates an inert WAV fallback; no reference audio or identity inference is accepted."""

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
            tts_model_revision="synthetic-fallback-v1",
            generation_settings={"sample_rate": sample_rate, "mode": "deterministic-tone"},
        )


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
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError("Install vishgym[audio] before loading Qwen3-TTS.") from exc
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        self._supported_speakers = set(self._model.get_supported_speakers())

    def render(self, team: Team, persona: Persona, text: str) -> AudioTurn:
        """Render a CustomVoice turn without accepting reference audio or arbitrary timbres."""
        if self._model is None:
            raise RuntimeError("Qwen3-TTS is not loaded.")
        if self._supported_speakers is not None and persona.voice_speaker not in self._supported_speakers:
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
