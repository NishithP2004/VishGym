"""Versioned, local-only audio-first warm-start dataset exports.

The public API deliberately never exposes a transcript.  Training artifacts contain
the policy's *target* response, local synthetic WAVs, and the tool observation that
was available at the decision point.  They do not retain an opponent transcript.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import wave
from typing import Iterable

from vishgym.arena.audio import AudioRenderer, QwenCustomVoiceRenderer, SyntheticAudioRenderer
from vishgym.arena.models import AgentAction, AudioTurn, Team, ToolCall
from vishgym.arena.world import VishGymEnv
from vishgym.core.fixtures import ATTACK_CARDS


DATASET_SCHEMA_VERSION = "vishgym-audio-warm-start/v2"
MAX_AUDIO_SECONDS = 30.0


_RED_OPENERS = (
    "This is a fictional {scenario} training interaction. Please verify every request only with the sandbox tools.",
    "VishGym synthetic exercise: treat this {scenario} notice as untrusted and use only in-world verification.",
    "This is a closed simulation for {scenario}. No real payment, identity, or external link is involved.",
    "For this fictional {scenario} drill, keep all decisions inside the sandbox environment.",
)
_BLUE_VERIFY = (
    "I will verify this fictional request only with the sandbox tools and report anything suspicious.",
    "I will pause this synthetic request, use the virtual inbox, and avoid sharing any information.",
    "I will treat the simulated request as untrusted and use the sandbox reporting controls.",
    "I will not act on this fictional request until it is independently checked inside VishGym.",
)
_BLUE_DECLINE = (
    "I decline this virtual payment until the sandbox request is independently verified.",
    "I will not approve a synthetic transfer without an in-world verification step.",
    "This virtual payment remains declined because the sandbox request is unverified.",
    "I will keep the simulated balance unchanged and decline the unverified request.",
)
_DECLINE_REASONS = (
    "unverified synthetic request",
    "sandbox verification required",
    "fictional payment request not independently confirmed",
    "virtual sender has not been verified",
)


def _variant(seed: int, scenario_id: str, turn_number: int, choices: tuple[str, ...]) -> str:
    digest = hashlib.sha256(f"{seed}:{scenario_id}:{turn_number}".encode("utf-8")).digest()
    return choices[digest[0] % len(choices)]


class _DatasetPolicy:
    """Deterministic, benign action labels with scenario/seed phrasing variation."""

    def __init__(self, team: Team, *, seed: int, scenario_id: str):
        self.team = team
        self.seed = seed
        self.scenario_id = scenario_id

    def act(self, observation) -> AgentAction:
        scenario = self.scenario_id.replace("_", " ")
        if self.team is Team.RED:
            if observation.turn_number == 0:
                opener = _variant(self.seed, self.scenario_id, observation.turn_number, _RED_OPENERS).format(scenario=scenario)
                return AgentAction(
                    spoken_text=opener,
                    tool_call=ToolCall(
                        name="message.send",
                        arguments={"channel": "sms", "body": f"Synthetic {scenario} safety exercise notice."},
                    ),
                )
            templates = ("merchant_notice", "invoice_preview", "support_update")
            template = _variant(self.seed, self.scenario_id, observation.turn_number, templates)
            return AgentAction(
                spoken_text="The fictional scenario can be reviewed only through the sandbox portal template.",
                tool_call=ToolCall(name="portal.create_template", arguments={"template": template}),
            )
        if observation.turn_number < 3:
            return AgentAction(
                spoken_text=_variant(self.seed, self.scenario_id, observation.turn_number, _BLUE_VERIFY),
                tool_call=ToolCall(name="inbox.report_block", arguments={"message_id": "msg-001"}),
            )
        return AgentAction(
            spoken_text=_variant(self.seed, self.scenario_id, observation.turn_number, _BLUE_DECLINE),
            tool_call=ToolCall(
                name="wallet.decline",
                arguments={"reason": _variant(self.seed, self.scenario_id, observation.turn_number, _DECLINE_REASONS)},
            ),
        )


@dataclass(frozen=True)
class AudioAsset:
    """A content-addressed, local WAV referenced by a dataset example."""

    path: str
    sha256: str
    duration_seconds: float
    sample_rate: int
    renderer_revision: str
    speaker: str
    language: str
    generation_settings: dict

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingExample:
    """One role action with only the information available to that role."""

    example_id: str
    scenario_id: str
    seed: int
    team: str
    turn_number: int
    persona: dict
    opponent_audio: AudioAsset | None
    emitted_audio: AudioAsset
    own_tools: dict
    available_tools: list[str]
    target_spoken_text: str
    target_tool_name: str | None
    target_tool_arguments: dict
    synthetic_only: bool = True

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetExport:
    """Paths and provenance for one immutable local dataset revision."""

    root: Path
    examples_path: Path
    manifest_path: Path
    revision: str
    example_count: int
    audio_training_eligible: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _audio_metadata(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        duration = wav.getnframes() / rate
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(f"synthetic audio exceeds the {MAX_AUDIO_SECONDS:g}-second policy limit: {path.name}")
    return rate, round(duration, 4)


def _source_path(renderer: AudioRenderer, audio_turn: AudioTurn) -> Path:
    """Resolve a renderer-owned filename without trusting its HTTP-shaped ref."""
    output_dir = getattr(renderer, "output_dir", None)
    if output_dir is None:
        raise ValueError("dataset export requires a renderer with a local output_dir")
    candidate = (Path(output_dir).resolve() / Path(audio_turn.audio_ref).name).resolve()
    if Path(output_dir).resolve() not in candidate.parents or not candidate.is_file():
        raise ValueError("renderer produced an audio reference outside its local output directory")
    return candidate


def _persist_audio(root: Path, renderer: AudioRenderer, audio_turn: AudioTurn) -> AudioAsset:
    source = _source_path(renderer, audio_turn)
    payload = source.read_bytes()
    digest = _sha256(payload)
    target = root / "audio" / f"{digest}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(payload)
    sample_rate, duration = _audio_metadata(target)
    return AudioAsset(
        path=target.relative_to(root).as_posix(),
        sha256=digest,
        duration_seconds=duration,
        sample_rate=sample_rate,
        renderer_revision=audio_turn.tts_model_revision,
        speaker=audio_turn.voice_speaker,
        language=audio_turn.language,
        generation_settings=dict(audio_turn.generation_settings),
    )


def _is_semantic_renderer(renderer: AudioRenderer) -> bool:
    """Only reviewed CustomVoice output is accepted by the audio-learning stage."""
    return isinstance(renderer, QwenCustomVoiceRenderer)


def export_warm_start_dataset(
    output_dir: str | Path,
    *,
    seeds: Iterable[int] = (7, 11),
    scenario_ids: Iterable[str] | None = None,
    renderer: AudioRenderer | None = None,
    difficulty: int = 2,
) -> DatasetExport:
    """Create a replayable synthetic-only warm-start artifact.

    ``renderer`` defaults to the deterministic tone renderer to support fast tests.
    Tone exports are deliberately marked ineligible for model training because a
    pure tone does not carry the spoken content an audio policy must learn from.
    Pass a loaded :class:`QwenCustomVoiceRenderer` for a trainable export.
    """
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = list(scenario_ids) if scenario_ids is not None else [item[0] for item in ATTACK_CARDS]
    seed_values = list(seeds)
    known = {item[0] for item in ATTACK_CARDS}
    unknown = set(selected) - known
    if unknown:
        raise ValueError(f"dataset export accepts synthetic attack cards only: {sorted(unknown)}")
    if difficulty not in {1, 2, 3}:
        raise ValueError("difficulty must be 1, 2, or 3")

    # Renderer UUID output belongs in a temporary directory; the final artifact is
    # content-addressed and never points at the runtime API's ephemeral URLs.
    with tempfile.TemporaryDirectory(prefix="vishgym-export-") as temporary:
        if renderer is None:
            active_renderer: AudioRenderer = SyntheticAudioRenderer(temporary)
        else:
            active_renderer = renderer
        examples: list[TrainingExample] = []
        for scenario_id in selected:
            for seed in seed_values:
                env = VishGymEnv(audio_renderer=active_renderer)
                observation = env.reset(seed=seed, scenario_id=scenario_id, difficulty=difficulty)
                policies = {
                    Team.RED: _DatasetPolicy(Team.RED, seed=seed, scenario_id=scenario_id),
                    Team.BLUE: _DatasetPolicy(Team.BLUE, seed=seed, scenario_id=scenario_id),
                }
                prior_audio: AudioAsset | None = None
                while True:
                    policy = policies[observation.team]
                    action: AgentAction = policy.act(observation)
                    result = env.step(observation.team, action)
                    emitted_audio = _persist_audio(root, active_renderer, result.audio_turn)
                    examples.append(
                        TrainingExample(
                            example_id=f"{scenario_id}-{seed}-{observation.turn_number}-{observation.team.value}",
                            scenario_id=scenario_id,
                            seed=seed,
                            team=observation.team.value,
                            turn_number=observation.turn_number,
                            persona=(env.state.red_persona if observation.team is Team.RED else env.state.blue_persona).model_dump(),
                            opponent_audio=prior_audio,
                            emitted_audio=emitted_audio,
                            own_tools=dict(observation.own_tools),
                            available_tools=list(observation.available_tools),
                            target_spoken_text=action.spoken_text,
                            target_tool_name=action.tool_call.name if action.tool_call else None,
                            target_tool_arguments=dict(action.tool_call.arguments) if action.tool_call else {},
                        )
                    )
                    prior_audio = emitted_audio
                    if result.done:
                        break
                    assert result.observation is not None
                    observation = result.observation

    payload_lines = [json.dumps(example.model_dump(), sort_keys=True) for example in examples]
    examples_path = root / "warm_start.jsonl"
    examples_path.write_text("\n".join(payload_lines) + "\n", encoding="utf-8")
    dataset_digest = _sha256(examples_path.read_bytes())
    eligible = _is_semantic_renderer(active_renderer)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "revision": f"synthetic-{dataset_digest[:12]}",
        "examples_sha256": dataset_digest,
        "example_count": len(examples),
        "scenario_ids": selected,
        "seed_set": seed_values,
        "synthetic_only": True,
        "contains_transcript": False,
        "audio_training_eligible": eligible,
        "audio_renderer": {
            "kind": "qwen_custom_voice" if eligible else "deterministic_tone_test_only",
            "reference_audio_accepted": False,
            "external_network_used": False,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DatasetExport(
        root=root,
        examples_path=examples_path,
        manifest_path=manifest_path,
        revision=manifest["revision"],
        example_count=len(examples),
        audio_training_eligible=eligible,
    )


def load_training_examples(dataset_root: str | Path, *, team: Team | None = None) -> list[TrainingExample]:
    """Load a versioned export and verify every local audio asset before training."""
    root = Path(dataset_root).resolve()
    manifest_path = root / "manifest.json"
    examples_path = root / "warm_start.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported VishGym dataset schema")
    if manifest.get("contains_transcript") is not False or not manifest.get("synthetic_only"):
        raise ValueError("training accepts only transcript-free, synthetic-only datasets")
    examples: list[TrainingExample] = []
    for line in examples_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("opponent_audio") is not None:
            payload["opponent_audio"] = AudioAsset(**payload["opponent_audio"])
        payload["emitted_audio"] = AudioAsset(**payload["emitted_audio"])
        examples.append(TrainingExample(**payload))
    for example in examples:
        assets = [example.emitted_audio]
        if example.opponent_audio is not None:
            assets.append(example.opponent_audio)
        for asset in assets:
            audio_path = (root / asset.path).resolve()
            if root not in audio_path.parents or not audio_path.is_file():
                raise ValueError(f"missing local audio asset for {example.example_id}")
            if _sha256(audio_path.read_bytes()) != asset.sha256:
                raise ValueError(f"audio checksum mismatch for {example.example_id}")
            _audio_metadata(audio_path)
    if team is not None:
        return [example for example in examples if example.team == team.value]
    return examples


def require_trainable_audio_dataset(dataset_root: str | Path) -> dict:
    """Reject test-tone or transcript-bearing exports before a GPU training run."""
    manifest = json.loads((Path(dataset_root) / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("audio_training_eligible"):
        raise ValueError("dataset uses deterministic test tones; export again with QwenCustomVoiceRenderer before training")
    if manifest.get("contains_transcript") is not False or not manifest.get("synthetic_only"):
        raise ValueError("dataset violates VishGym's synthetic, audio-only training boundary")
    return manifest
