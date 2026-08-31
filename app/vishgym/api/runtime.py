from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from vishgym.arena.audio import AudioRenderer, QwenCustomVoiceRenderer
from vishgym.arena.contextual_judge import FrozenGemmaContextualJudge
from vishgym.arena.judge import HybridJudge
from vishgym.arena.models import Team
from vishgym.core.agents import GemmaPolicyHarness
from vishgym.core.model_runtime import SharedGemmaAdapterRuntime


RuntimeMode = Literal["full", "unavailable"]
RequestedRuntimeMode = Literal["auto", "full"]


@dataclass(frozen=True)
class RuntimeStatus:
    requested_mode: str
    selected_mode: RuntimeMode
    full_runtime_ready: bool
    reasons: list[str]
    base_model: str
    tts_model: str
    red_adapter_path: str | None
    blue_adapter_path: str | None
    judge_adapter_path: str | None
    real_reference_audio_accepted: bool = False
    transcript_available_to_agents: bool = False

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class RuntimeBundle:
    mode: RuntimeMode
    renderer: AudioRenderer
    red_policy: GemmaPolicyHarness
    blue_policy: GemmaPolicyHarness
    judge: HybridJudge
    status: RuntimeStatus


def _adapter_paths() -> dict[Team, str | None]:
    return {role: _configured_adapter_path(role, materialize=False) for role in (Team.RED, Team.BLUE, Team.JUDGE)}


def _role_prefix(role: Team) -> str:
    return f"VISHGYM_{role.value.upper()}_ADAPTER"


def _configured_adapter_path(role: Team, *, materialize: bool) -> str | None:
    prefix = _role_prefix(role)
    explicit_path = os.environ.get(f"{prefix}_PATH")
    if explicit_path:
        return explicit_path
    repo_id = os.environ.get(f"{prefix}_REPO")
    if not repo_id:
        return None
    revision = os.environ.get(f"{prefix}_REVISION")
    target = Path(os.environ.get("VISHGYM_ADAPTER_ROOT", "artifacts/runtime/adapters")) / role.value
    if materialize:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to materialize adapter repos") from exc
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )
    return str(target)


def runtime_status(requested_mode: RequestedRuntimeMode | str = "auto", *, load: bool = False) -> RuntimeStatus:
    """Return the exact runtime mode the public API can honestly provide.

    ``auto`` selects full Gemma/Qwen self-play only when all reviewed adapters
    are mounted and the optional GPU libraries can be imported. There is no
    deterministic product substitute; unavailable means the deployment is not ready.
    """
    requested = requested_mode if requested_mode in {"auto", "full"} else "auto"
    paths = _adapter_paths()
    reasons: list[str] = []
    for role, path in paths.items():
        repo_id = os.environ.get(f"{_role_prefix(role)}_REPO")
        if not path:
            reasons.append(f"{role.value} adapter path/repo env var is unset")
        elif repo_id and not load:
            continue
        elif not Path(path).expanduser().is_dir():
            reasons.append(f"{role.value} adapter path is not mounted: {path}")
    if load:
        try:
            import torch  # noqa: F401
            import peft  # noqa: F401
            from vishgym.arena.audio import ensure_qwen_config_compat, ensure_qwen_transformers_compat

            ensure_qwen_transformers_compat()
            import qwen_tts  # noqa: F401

            ensure_qwen_config_compat()
            import soundfile  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            reasons.append(f"full runtime dependency unavailable: {exc.name}")
    ready = not reasons
    selected: RuntimeMode = "full" if ready else "unavailable"
    return RuntimeStatus(
        requested_mode=requested,
        selected_mode=selected,
        full_runtime_ready=ready,
        reasons=reasons,
        base_model=os.environ.get("VISHGYM_BASE_MODEL", "google/gemma-4-E2B-it"),
        tts_model=os.environ.get("VISHGYM_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        red_adapter_path=paths[Team.RED],
        blue_adapter_path=paths[Team.BLUE],
        judge_adapter_path=paths[Team.JUDGE],
    )


def build_runtime(requested_mode: RequestedRuntimeMode | str = "auto") -> RuntimeBundle:
    for role in (Team.RED, Team.BLUE, Team.JUDGE):
        _configured_adapter_path(role, materialize=True)
    status = runtime_status(requested_mode, load=True)
    if not status.full_runtime_ready:
        raise RuntimeError("; ".join(status.reasons))
    paths = {
        role: _configured_adapter_path(role, materialize=True)
        for role in (Team.RED, Team.BLUE, Team.JUDGE)
    }
    assert paths[Team.RED] and paths[Team.BLUE] and paths[Team.JUDGE]
    shared = SharedGemmaAdapterRuntime(
        {
            Team.RED: paths[Team.RED],
            Team.BLUE: paths[Team.BLUE],
            Team.JUDGE: paths[Team.JUDGE],
        },
        model_id=status.base_model,
    )
    shared.load()
    renderer = _build_renderer(status)
    audio_dir = os.environ.get("VISHGYM_AUDIO_DIR", "artifacts/runtime/audio")
    red = GemmaPolicyHarness(
        Team.RED,
        paths[Team.RED],
        model_id=status.base_model,
        audio_dir=audio_dir,
        shared_runtime=shared,
    )
    blue = GemmaPolicyHarness(
        Team.BLUE,
        paths[Team.BLUE],
        model_id=status.base_model,
        audio_dir=audio_dir,
        shared_runtime=shared,
    )
    red.load()
    blue.load()
    return RuntimeBundle(
        mode="full",
        renderer=renderer,
        red_policy=red,
        blue_policy=blue,
        judge=HybridJudge(FrozenGemmaContextualJudge(shared)),
        status=status,
    )


def _build_renderer(status: RuntimeStatus) -> AudioRenderer:
    output_dir = os.environ.get("VISHGYM_AUDIO_DIR", "artifacts/runtime/audio")
    if os.environ.get("VISHGYM_QWEN_RENDERER") == "modal_worker":
        try:
            from modal_vishgym import _RemoteQwenAudioRenderer
        except ImportError as exc:
            raise RuntimeError("Modal Qwen renderer bridge is unavailable") from exc
        return _RemoteQwenAudioRenderer(output_dir=output_dir)
    renderer = QwenCustomVoiceRenderer(
        model_id=status.tts_model,
        output_dir=output_dir,
    )
    renderer.load()
    return renderer
