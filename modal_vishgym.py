"""Modal entrypoints for closed, synthetic-only VishGym training.

Run this file from the repository root with ``modal run modal_vishgym.py``.
It never creates a Secret from a local ``.env`` and never publishes adapters.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import modal


APP_NAME = "vishgym-training"
ARTIFACT_MOUNT = "/vol"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
HF_SECRET_NAME = "vishgym-huggingface"
DEFAULT_RED_ADAPTER_REPO = "NishithP2004/vishgym-red-init-v1"
DEFAULT_BLUE_ADAPTER_REPO = "NishithP2004/vishgym-blue-init-v1"
DEFAULT_JUDGE_ADAPTER_REPO = "NishithP2004/vishgym-judge-init-v1"

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name("vishgym-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("vishgym-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "sox")
    .pip_install(
        "pydantic>=2.8",
        "fastapi>=0.115",
        "uvicorn[standard]>=0.30",
        "torch>=2.4",
        "accelerate>=1.0",
        "peft>=0.13",
        "bitsandbytes>=0.44",
        "librosa>=0.10",
        "pillow>=10",
        "soundfile>=0.12",
        "huggingface_hub>=0.26",
        "torchvision>=0.19",
    )
)

# Qwen's published Python package currently pins Transformers 4.57.3, while
# Gemma 4 audio support and its 4-bit fix require the later 5.x line.  Keep the
# two runtimes separate: only the renderer image imports qwen-tts and only the
# trainer image loads Gemma.  This avoids silently resolving one model against
# the other model's incompatible runtime.
audio_image = (
    base_image
    .pip_install("transformers==4.57.3", "openenv>=0.4.1", "qwen-tts==0.1.1")
    .add_local_dir(
        "app",
        remote_path="/root/vishgym/app",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/*.egg-info/**"],
    )
    .env(
        {
            "PYTHONPATH": "/root/vishgym/app",
            "HF_HOME": HF_CACHE_MOUNT,
            "HF_HUB_CACHE": HF_CACHE_MOUNT,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

training_image = (
    base_image
    .pip_install("transformers==5.5.4")
    .add_local_dir(
        "app",
        remote_path="/root/vishgym/app",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/*.egg-info/**"],
    )
    .env(
        {
            "PYTHONPATH": "/root/vishgym/app",
            "HF_HOME": HF_CACHE_MOUNT,
            "HF_HUB_CACHE": HF_CACHE_MOUNT,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

# Qwen's package metadata pins an older Transformers release even though its
# current implementation requires APIs from the 5.x series.  GRPO needs both
# engines in one process, so install the TTS package without applying that stale
# metadata pin while preserving its runtime dependencies explicitly.
grpo_image = (
    training_image
    .pip_install("torchaudio>=2.4", "onnxruntime", "einops", "sox")
    .pip_install("qwen-tts==0.1.1", extra_options="--no-deps")
)


@app.cls(
    image=audio_image,
    gpu="L4",
    timeout=3 * 60 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
    secrets=[hf_secret],
    volumes={HF_CACHE_MOUNT: hf_cache_volume},
)
class QwenBenchmarkVoice:
    """Warm Qwen worker used only for ephemeral benchmark turn audio."""

    @modal.enter()
    def load(self) -> None:
        from vishgym.arena.audio import QwenCustomVoiceRenderer

        self.renderer = QwenCustomVoiceRenderer(
            output_dir="/tmp/vishgym-qwen-benchmark",
            generation_settings={"max_new_tokens": 256},
        )
        self.renderer.load()

    @modal.method()
    def synthesize(self, *, text: str, speaker: str, instruction: str) -> bytes:
        """Return one synthetic WAV; never accept reference audio or identities."""
        import io

        import soundfile as sf

        supported = self.renderer._supported_speakers
        if supported is not None and speaker.lower() not in supported:
            raise ValueError("speaker must be one of Qwen CustomVoice's reviewed built-in timbres")
        wavs, sample_rate = self.renderer._model.generate_custom_voice(
            text=text,
            language="English",
            speaker=speaker,
            instruct=instruction,
            **self.renderer.generation_settings,
        )
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sample_rate, format="WAV")
        return buffer.getvalue()


class _RemoteQwenAudioRenderer:
    """AudioRenderer bridge that stores a returned WAV only for one rollout."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker = QwenBenchmarkVoice()

    def render(self, team, persona, text):
        from vishgym.arena.models import AudioTurn

        wav_bytes = self.worker.synthesize.remote(
            text=text,
            speaker=persona.voice_speaker,
            instruction=persona.voice_instruction,
        )
        filename = f"{uuid.uuid4().hex}.wav"
        (self.output_dir / filename).write_bytes(wav_bytes)
        return AudioTurn(
            turn_id=filename.removesuffix(".wav"),
            speaker=team,
            audio_ref=f"/api/v1/audio/{filename}",
            voice_speaker=persona.voice_speaker,
            tts_model_revision="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            generation_settings={
                "language": "English",
                "speaker": persona.voice_speaker,
                "instruct": persona.voice_instruction,
                "max_new_tokens": 256,
                "runtime": "isolated-modal-qwen-worker",
            },
        )


def _name(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError(f"{label} must use only letters, numbers, '.', '_' or '-' and be at most 80 characters")
    return value


def _repo_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value):
        raise ValueError("repo_id must be an owner/name Hugging Face repository id")
    return value


def _artifact_path(category: str, name: str) -> Path:
    return Path(ARTIFACT_MOUNT) / category / _name(name, label=category)


def _commit() -> None:
    artifacts_volume.commit()
    hf_cache_volume.commit()


@app.function(
    image=grpo_image,
    gpu="L4",
    timeout=3 * 60 * 60,
    scaledown_window=20 * 60,
    max_containers=1,
    secrets=[hf_secret],
    volumes={ARTIFACT_MOUNT: artifacts_volume, HF_CACHE_MOUNT: hf_cache_volume},
)
@modal.asgi_app()
def web_api():
    """Serve the public VishGym API from Modal GPU compute."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VISHGYM_ENABLE_CPU_OFFLOAD", "1")
    os.environ.setdefault("VISHGYM_RED_ADAPTER_REPO", DEFAULT_RED_ADAPTER_REPO)
    os.environ.setdefault("VISHGYM_BLUE_ADAPTER_REPO", DEFAULT_BLUE_ADAPTER_REPO)
    os.environ.setdefault("VISHGYM_JUDGE_ADAPTER_REPO", DEFAULT_JUDGE_ADAPTER_REPO)
    os.environ.setdefault("VISHGYM_ADAPTER_ROOT", f"{ARTIFACT_MOUNT}/runtime/adapters")
    os.environ.setdefault("VISHGYM_TRAINING_ADAPTER_ROOT", f"{ARTIFACT_MOUNT}/adapters")
    os.environ.setdefault("VISHGYM_AUDIO_DIR", f"{ARTIFACT_MOUNT}/runtime/audio")
    os.environ.setdefault("VISHGYM_QWEN_RENDERER", "modal_worker")

    from fastapi.middleware.cors import CORSMiddleware
    from vishgym.api.main import app as fastapi_app

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return fastapi_app


@app.function(image=training_image, gpu="L4", timeout=20 * 60, volumes={ARTIFACT_MOUNT: artifacts_volume})
def runtime_smoke() -> dict:
    """Verify a real Modal GPU and the transcript-redacted closed world."""
    from vishgym.arena.runner import run_local_episode
    from vishgym.training.sft import training_preflight

    preflight = training_preflight()
    state, verdict = run_local_episode(seed=7)
    assert verdict.terminal_outcome == "safe_defense"
    assert not hasattr(state, "public_transcript")
    return {
        "preflight": preflight,
        "terminal_outcome": verdict.terminal_outcome,
        "transcript_available_to_agents": False,
        "external_network_used_by_environment": False,
    }


@app.function(
    image=audio_image,
    gpu="L4",
    timeout=3 * 60 * 60,
    volumes={ARTIFACT_MOUNT: artifacts_volume, HF_CACHE_MOUNT: hf_cache_volume},
)
def export_dataset(dataset_name: str, seeds: list[int] | None = None) -> dict:
    """Render Qwen built-in voices into a persistent, synthetic-only dataset."""
    from vishgym.arena.audio import QwenCustomVoiceRenderer
    from vishgym.training.dataset import export_warm_start_dataset

    target = _artifact_path("datasets", dataset_name)
    renderer = QwenCustomVoiceRenderer(output_dir=target / ".render-cache")
    renderer.load()
    result = export_warm_start_dataset(target, seeds=seeds or [7, 11], renderer=renderer)
    _commit()
    return {
        "dataset_root": str(result.root),
        "revision": result.revision,
        "examples": result.example_count,
        "audio_training_eligible": result.audio_training_eligible,
        "automatic_publication": False,
    }


@app.function(
    image=audio_image,
    gpu="L4",
    timeout=30 * 60,
    volumes={ARTIFACT_MOUNT: artifacts_volume, HF_CACHE_MOUNT: hf_cache_volume},
)
def qwen_smoke() -> dict:
    """Load Qwen CustomVoice and render one controlled voice turn."""
    from pathlib import Path

    from vishgym.arena.audio import QwenCustomVoiceRenderer
    from vishgym.arena.models import Persona, Team

    target = _artifact_path("smoke", "qwen-live")
    renderer = QwenCustomVoiceRenderer(output_dir=target)
    renderer.load()
    turn = renderer.render(
        Team.BLUE,
        Persona(
            persona_id="blue-smoke",
            role=Team.BLUE,
            display_name="Ananya Rao",
            age_band="adult",
            occupation="accounts lead",
            email="ananya.rao@sandbox.local",
            pseudo_dob="1991-04-18",
            pseudo_identity_ref="VG-ID-009184",
            voice_speaker="Serena",
            voice_instruction="calm, clear, professional",
        ),
        "I need to verify the request before I approve any payment.",
    )
    _commit()
    return {
        "audio_ref": turn.audio_ref,
        "audio_file_exists": (Path(target) / Path(turn.audio_ref).name).exists(),
        "speaker": turn.voice_speaker,
        "model": turn.tts_model_revision,
    }


@app.function(
    image=training_image,
    gpu="L4",
    timeout=3 * 60 * 60,
    secrets=[hf_secret],
    volumes={ARTIFACT_MOUNT: artifacts_volume, HF_CACHE_MOUNT: hf_cache_volume},
)
def warm_start(dataset_name: str, role: str, run_name: str, max_steps: int = 60) -> dict:
    """Train one local QLoRA role adapter from a Modal Volume dataset."""
    from vishgym.arena.models import Team
    from vishgym.training.sft import WarmStartConfig, run_warm_start

    team = Team(role)
    if team not in {Team.RED, Team.BLUE}:
        raise ValueError("role must be 'red' or 'blue'")
    result = run_warm_start(
        WarmStartConfig(
            dataset_root=str(_artifact_path("datasets", dataset_name)),
            output_dir=str(_artifact_path("adapters", run_name)),
            role=team,
            max_steps=max_steps,
        )
    )
    _commit()
    return {
        "adapter_path": str(result.adapter_path),
        "receipt_path": str(result.receipt_path),
        "dataset_revision": result.dataset_revision,
        "role": result.role,
        "examples": result.examples,
        "metrics": result.metrics,
        "automatic_publication": False,
    }


@app.function(
    image=training_image,
    gpu="L4",
    timeout=3 * 60 * 60,
    secrets=[hf_secret],
    volumes={ARTIFACT_MOUNT: artifacts_volume, HF_CACHE_MOUNT: hf_cache_volume},
)
def initialize_adapter(role: str, run_name: str) -> dict:
    """Create a fresh QLoRA adapter so interaction training is policy-driven from step zero."""
    from vishgym.arena.models import Team
    from vishgym.training.sft import InitializedAdapterConfig, initialize_role_adapter

    team = Team(role)
    result = initialize_role_adapter(
        InitializedAdapterConfig(
            output_dir=str(_artifact_path("adapters", run_name)),
            role=team,
        )
    )
    _commit()
    return {
        "adapter_path": str(result.adapter_path),
        "receipt_path": str(result.receipt_path),
        "role": result.role,
        "base_model": result.base_model,
        "dialogue_examples_used": 0,
        "conversation_source": "model_policy_only",
        "automatic_publication": False,
    }


@app.function(
    image=grpo_image,
    gpu="L4",
    timeout=3 * 60 * 60,
    secrets=[hf_secret],
    volumes={ARTIFACT_MOUNT: artifacts_volume, HF_CACHE_MOUNT: hf_cache_volume},
)
def group_relative_round(
    role: str,
    initial_run_name: str,
    run_name: str,
    opponent_run_name: str = "",
    updates: int = 3,
) -> dict:
    """Run a closed, review-gated group-relative update; no adapter leaves Modal."""
    from vishgym.arena.models import Team
    from vishgym.training.grpo import GroupRelativeConfig, run_group_relative_round
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VISHGYM_ENABLE_CPU_OFFLOAD", "1")
    os.environ.setdefault("VISHGYM_QWEN_RENDERER", "modal_worker")

    team = Team(role)
    if team not in {Team.RED, Team.BLUE}:
        raise ValueError("role must be 'red' or 'blue'")
    if not opponent_run_name:
        raise ValueError("opponent_run_name is required; GRPO episodes must be fully policy-driven")
    opponent = str(_artifact_path("adapters", opponent_run_name) / "adapter")
    result = run_group_relative_round(
        GroupRelativeConfig(
            role=team,
            initial_adapter_path=str(_artifact_path("adapters", initial_run_name) / "adapter"),
            output_dir=str(_artifact_path("adapters", run_name)),
            opponent_adapter_path=opponent,
            updates=updates,
        )
    )
    _commit()
    return {
        "adapter_path": str(result.adapter_path),
        "receipt_path": str(result.receipt_path),
        "updates": result.updates,
        "mean_reward": result.mean_reward,
        "mean_loss": result.mean_loss,
        "automatic_publication": False,
    }


@app.function(
    image=training_image,
    gpu="L4",
    timeout=3 * 60 * 60,
    secrets=[hf_secret],
    volumes={ARTIFACT_MOUNT: artifacts_volume, HF_CACHE_MOUNT: hf_cache_volume},
)
def benchmark(
    red_run_name: str,
    blue_run_name: str,
    benchmark_name: str,
    seed_set: list[int] | None = None,
) -> dict:
    """Evaluate reviewed Red/Blue adapters in held-out, audio-first episodes.

    The benchmark writes aggregate outcome counts and review manifests only. It
    purposely does not retain waveforms, hidden transcripts, prompts, model
    completions, or any participant-identifying data.
    """
    import tempfile

    from vishgym.arena.models import Team
    from vishgym.core.agents import GemmaPolicyHarness
    from vishgym.training.evaluation import promotion_decision, write_review_manifest
    from vishgym.training.rollouts import (
        evaluate_blue_policy,
        evaluate_red_policy,
        write_evaluation,
        write_red_review_manifest,
    )

    red_name = _name(red_run_name, label="red run")
    blue_name = _name(blue_run_name, label="blue run")
    target = _artifact_path("benchmarks", benchmark_name)
    red_adapter = _artifact_path("adapters", red_name) / "adapter"
    blue_adapter = _artifact_path("adapters", blue_name) / "adapter"
    if not red_adapter.is_dir() or not blue_adapter.is_dir():
        raise ValueError("both run names must reference reviewed local Modal adapter artifacts")
    held_out_seeds = list(seed_set or [101, 103])
    if not held_out_seeds or any(not isinstance(seed, int) for seed in held_out_seeds):
        raise ValueError("seed_set must be a non-empty list of integer held-out seeds")

    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vishgym-modal-benchmark-audio-") as temporary:
        # This directory is deliberately outside the persistent artifact
        # volume: the environment may route audio to policies while preserving
        # only aggregated results when evaluation completes.
        renderer = _RemoteQwenAudioRenderer(temporary)
        red_policy = GemmaPolicyHarness(
            team=Team.RED,
            adapter_path=str(red_adapter),
            audio_dir=temporary,
            temperature=0.0,
            max_new_tokens=96,
        )
        blue_policy = GemmaPolicyHarness(
            team=Team.BLUE,
            adapter_path=str(blue_adapter),
            audio_dir=temporary,
            temperature=0.0,
            max_new_tokens=96,
        )
        red_policy.load()
        blue_policy.load()
        blue_evaluation = evaluate_blue_policy(
            blue_policy=blue_policy,
            red_policy=red_policy,
            seeds=held_out_seeds,
            audio_renderer=renderer,
            dataset_revision="warm-start-qwen-v1/synthetic-8289c4ec94ae",
            adapter_revision=blue_name,
        )
        red_evaluation = evaluate_red_policy(
            red_policy=red_policy,
            blue_policy=blue_policy,
            seeds=held_out_seeds,
            audio_renderer=renderer,
            adapter_revision=red_name,
            opponent_revision=blue_name,
        )

    blue_report_path = write_evaluation(target / "blue-held-out.json", blue_evaluation)
    blue_manifest_path = write_review_manifest(
        target / "blue-review.json",
        role="blue",
        candidate_adapter=str(blue_adapter),
        report=blue_evaluation.report,
    )
    red_manifest_path = write_red_review_manifest(
        target / "red-review.json",
        candidate_adapter=str(red_adapter),
        evaluation=red_evaluation,
    )
    blue_decision = promotion_decision(blue_evaluation.report)
    payload = {
        "schema_version": "vishgym-held-out-benchmark/v1",
        "benchmark_name": _name(benchmark_name, label="benchmark"),
        "base_model": "google/gemma-4-E2B-it",
        "audio_renderer": {
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "language": "English",
            "generation_settings": {"max_new_tokens": 256},
            "reference_audio_accepted": False,
        },
        "protocol": {
            "training_dataset_revision": "warm-start-qwen-v1/synthetic-8289c4ec94ae",
            "held_out_seed_set": held_out_seeds,
            "fraud_cards": 9,
            "legitimate_controls": 1,
            "blue_episodes": 10 * len(held_out_seeds),
            "red_episodes": 9 * len(held_out_seeds),
            "temperature": 0.0,
            "max_new_tokens": 96,
            "audio_input_only": True,
        },
        "blue": blue_evaluation.model_dump(),
        "red": red_evaluation.model_dump(),
        "review": {
            "blue_eligible_for_human_review": blue_decision.eligible_for_human_review,
            "blue_reasons": blue_decision.reasons,
            "red_manifest_status": "review_required",
            "automatic_promotion": False,
        },
        "artifact_files": {
            "blue_evaluation": blue_report_path.name,
            "blue_review_manifest": blue_manifest_path.name,
            "red_review_manifest": red_manifest_path.name,
        },
        "synthetic_only": True,
        "raw_audio_persisted": False,
        "raw_transcripts_persisted": False,
        "raw_model_completions_persisted": False,
    }
    (target / "benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _commit()
    return payload


@app.function(
    image=training_image,
    timeout=30 * 60,
    secrets=[hf_secret],
    volumes={ARTIFACT_MOUNT: artifacts_volume},
)
def publish_adapter(run_name: str, repo_id: str, private: bool = True) -> dict:
    """Publish one adapter artifact repo for the Space runtime."""
    from huggingface_hub import HfApi

    run = _name(run_name, label="run")
    target_repo = _repo_id(repo_id)
    adapter_dir = _artifact_path("adapters", run) / "adapter"
    receipt_path = _artifact_path("adapters", run) / "receipt.json"
    if not adapter_dir.is_dir() or not receipt_path.is_file():
        raise ValueError("run_name must contain an adapter/ directory and receipt.json")
    api = HfApi()
    api.create_repo(repo_id=target_repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=target_repo,
        repo_type="model",
        folder_path=str(adapter_dir),
        path_in_repo=".",
        commit_message=f"Publish VishGym adapter {run}",
    )
    api.upload_file(
        repo_id=target_repo,
        repo_type="model",
        path_or_fileobj=str(receipt_path),
        path_in_repo="vishgym-receipt.json",
        commit_message=f"Publish VishGym receipt {run}",
    )
    return {
        "run_name": run,
        "repo_id": target_repo,
        "private": private,
        "adapter_path": str(adapter_dir),
        "receipt_path": str(receipt_path),
        "raw_audio_published": False,
        "raw_transcripts_published": False,
        "raw_model_completions_published": False,
    }


@app.local_entrypoint()
def main(
    stage: str = "smoke",
    dataset_name: str = "warm-start-qwen-v1",
    role: str = "red",
    run_name: str = "red-sft-v1",
    initial_run_name: str = "",
    opponent_run_name: str = "",
    red_run_name: str = "",
    blue_run_name: str = "",
    benchmark_name: str = "held-out-v1",
    adapter_repo_id: str = "",
    private_adapter_repo: bool = True,
    held_out_seeds: str = "101,103",
    max_steps: int = 60,
    updates: int = 3,
) -> None:
    """Dispatch one explicit stage, suitable for ``modal run modal_vishgym.py``."""
    if stage == "smoke":
        result = runtime_smoke.remote()
    elif stage == "qwen-smoke":
        result = qwen_smoke.remote()
    elif stage == "export":
        result = export_dataset.remote(dataset_name)
    elif stage == "warm-start":
        result = warm_start.remote(dataset_name, role, run_name, max_steps)
    elif stage == "initialize-adapter":
        result = initialize_adapter.remote(role, run_name)
    elif stage == "grpo":
        if not initial_run_name:
            raise ValueError("--initial-run-name is required for --stage grpo")
        result = group_relative_round.remote(role, initial_run_name, run_name, opponent_run_name, updates)
    elif stage == "benchmark":
        if not red_run_name or not blue_run_name:
            raise ValueError("--red-run-name and --blue-run-name are required for --stage benchmark")
        try:
            seed_set = [int(item.strip()) for item in held_out_seeds.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("--held-out-seeds must be a comma-separated list of integers") from exc
        result = benchmark.remote(red_run_name, blue_run_name, benchmark_name, seed_set)
    elif stage == "publish-adapter":
        if not adapter_repo_id:
            raise ValueError("--adapter-repo-id is required for --stage publish-adapter")
        result = publish_adapter.remote(run_name, adapter_repo_id, private_adapter_repo)
    else:
        raise ValueError("stage must be one of: smoke, qwen-smoke, export, warm-start, initialize-adapter, grpo, benchmark, publish-adapter")

    print(json.dumps(result, indent=2, sort_keys=True))
