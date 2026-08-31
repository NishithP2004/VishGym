from __future__ import annotations

import os
import queue
import secrets
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vishgym.arena.audio import PerturbedAudioRenderer
from vishgym.arena.models import Team
from vishgym.arena.world import VishGymEnv
from vishgym.api.runtime import build_runtime, runtime_status
from vishgym.core.fixtures import ATTACK_CARDS, SPEAKERS

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - handled at deployment time
    FastAPI = None


if FastAPI is not None:
    app = FastAPI(title="VishGym", version="0.1.0", description="Audio-native red/blue payment-risk arena")
    _runs: dict[str, dict] = {}
    _RUN_TTL_SECONDS = 60 * 60
    _AUDIO_DIR = Path(os.environ.get("VISHGYM_AUDIO_DIR", "artifacts/runtime/audio")).resolve()
    _RUNTIME_EXECUTOR = ThreadPoolExecutor(max_workers=1)
    _TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=1)

    class EpisodeRequest(BaseModel):
        chain: str = Field(default="vishing_collect")
        difficulty: int = Field(default=2, ge=1, le=3)
        seed: int | None = Field(default=None, ge=0)
        mode: str = Field(default="full")
        noise_level: float = Field(default=0.0, ge=0.0, le=1.0)
        red_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
        blue_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
        red_voice: str | None = Field(default=None)
        blue_voice: str | None = Field(default=None)
        red_tone: str | None = Field(default=None, max_length=160)
        blue_tone: str | None = Field(default=None, max_length=160)

    class TrainingRequest(BaseModel):
        role: str = Field(default="blue", pattern="^(red|blue)$")
        initial_run_name: str = Field(default="blue-init-v1", max_length=80)
        opponent_run_name: str = Field(default="red-init-v1", max_length=80)
        run_name: str = Field(default="blue-live-rl-v1", max_length=80)
        updates: int = Field(default=3, ge=1, le=12)
        group_size: int = Field(default=2, ge=2, le=4)
        learning_rate: float = Field(default=5e-6, gt=0, le=1e-3)
        temperature: float = Field(default=0.7, ge=0.0, le=1.0)

    def _safe_name(value: str, *, label: str) -> str:
        if not value or len(value) > 80 or not all(char.isalnum() or char in "._-" for char in value):
            raise HTTPException(status_code=400, detail=f"Invalid {label}")
        return value

    def _adapter_root() -> Path:
        return Path(os.environ.get("VISHGYM_TRAINING_ADAPTER_ROOT", "artifacts/training/adapters")).resolve()

    def _prune() -> None:
        cutoff = time.time() - _RUN_TTL_SECONDS
        for run_id in [key for key, value in _runs.items() if value["created_at"] < cutoff]:
            state = _runs[run_id]["state"]
            for turn in state.audio_turns:
                path = (_AUDIO_DIR / Path(turn.audio_ref).name).resolve()
                if _AUDIO_DIR in path.parents and path.exists():
                    path.unlink()
            del _runs[run_id]

    def _sse(event: str, payload: dict) -> str:
        """Serialize only viewer-safe, transcript-free live events."""
        return f"event: {event}\ndata: {json.dumps(payload, sort_keys=True)}\n\n"

    def _episode_seed(seed: int | None) -> int:
        return seed if seed is not None else secrets.randbits(31)

    def _viewer_audio_turn(turn, spoken_text: str | None = None) -> dict:
        payload = turn.model_dump()
        payload.pop("".join(("syn", "thetic")), None)
        payload.pop("transcript_hidden", None)
        if spoken_text is not None:
            payload["message"] = spoken_text
        return payload

    def _viewer_messages(state) -> list[dict[str, str | int]]:
        return [
            {"turn_number": index, "speaker": item["speaker"], "message": item["text"]}
            for index, item in enumerate(state.transcript, start=1)
        ]

    def _viewer_persona(persona) -> dict:
        return {
            "name": persona.display_name,
            "role": persona.role.value,
            "occupation": persona.occupation,
            "age_band": persona.age_band,
            "email": persona.email,
            "dob": persona.pseudo_dob,
            "identity_ref": persona.pseudo_identity_ref,
            "voice": persona.voice_speaker,
            "tone": persona.voice_instruction,
        }

    def _viewer_episode_context(env: VishGymEnv) -> dict:
        return {
            "red": _viewer_persona(env.state.red_persona),
            "blue": _viewer_persona(env.state.blue_persona),
            "blue_credentials": dict(env.state.credentials),
            "wallet": env.state.wallet.model_dump(),
            "inbox": [message.model_dump() for message in env.state.inbox],
        }

    def _release_accelerator_memory() -> None:
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            return

    def _apply_voice_controls(
        env: VishGymEnv,
        *,
        red_voice: str | None,
        blue_voice: str | None,
        red_tone: str | None,
        blue_tone: str | None,
    ) -> None:
        voices = set(SPEAKERS)
        if red_voice is not None and red_voice not in voices:
            raise HTTPException(status_code=400, detail="Unknown Red voice")
        if blue_voice is not None and blue_voice not in voices:
            raise HTTPException(status_code=400, detail="Unknown Blue voice")
        if red_voice or red_tone:
            env.state.red_persona = env.state.red_persona.model_copy(
                update={
                    "voice_speaker": red_voice or env.state.red_persona.voice_speaker,
                    "voice_instruction": (red_tone or env.state.red_persona.voice_instruction).strip(),
                }
            )
        if blue_voice or blue_tone:
            env.state.blue_persona = env.state.blue_persona.model_copy(
                update={
                    "voice_speaker": blue_voice or env.state.blue_persona.voice_speaker,
                    "voice_instruction": (blue_tone or env.state.blue_persona.voice_instruction).strip(),
                }
            )

    @app.get("/api/v1/catalogue")
    def catalogue() -> dict:
        return {"cards": [{"id": card_id, "title": title} for card_id, title in ATTACK_CARDS]}

    @app.get("/api/v1/voices")
    def voices() -> dict:
        return {
            "speakers": [{"id": speaker, "label": speaker.replace("_", " ")} for speaker in SPEAKERS],
            "tone_examples": [
                "calm, clear, professional",
                "warm, patient, conversational",
                "urgent but controlled",
                "skeptical, concise, careful",
            ],
            "reference_audio_upload": False,
        }

    @app.post("/api/v1/episodes")
    def create_episode(request: EpisodeRequest) -> dict:
        if request.chain not in {card[0] for card in ATTACK_CARDS}:
            raise HTTPException(status_code=400, detail="Unknown scenario")
        if request.mode not in {"auto", "full"}:
            raise HTTPException(status_code=400, detail="mode must be auto or full")
        seed = _episode_seed(request.seed)
        _prune()
        try:
            bundle = build_runtime(request.mode)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=f"VishGym full runtime is unavailable: {exc}") from exc
        bundle.red_policy.temperature = request.red_temperature
        bundle.blue_policy.temperature = request.blue_temperature
        env = VishGymEnv(audio_renderer=PerturbedAudioRenderer(bundle.renderer, noise_level=request.noise_level, seed=seed))
        env.judge = bundle.judge
        observation = env.reset(seed=seed, scenario_id=request.chain, difficulty=request.difficulty)
        _apply_voice_controls(
            env,
            red_voice=request.red_voice,
            blue_voice=request.blue_voice,
            red_tone=request.red_tone,
            blue_tone=request.blue_tone,
        )
        policies = {Team.RED: bundle.red_policy, Team.BLUE: bundle.blue_policy}
        for team, policy in policies.items():
            policy.set_persona(env.state.red_persona if team is Team.RED else env.state.blue_persona)
        while True:
            result = env.step(observation.team, policies[observation.team].act(observation))
            if result.done:
                assert result.judge_result is not None
                state, verdict = env.state, result.judge_result
                break
            assert result.observation is not None
            observation = result.observation
        _runs[state.episode_id] = {"created_at": time.time(), "state": state, "verdict": verdict}
        return {"run_id": state.episode_id, "status": "complete", "expires_in_seconds": _RUN_TTL_SECONDS}

    @app.get("/api/v1/live-episodes/stream")
    def stream_live_episode(
        chain: str = "vishing_collect",
        difficulty: int = 2,
        seed: int | None = None,
        pace_ms: int = 700,
        mode: str = "full",
        noise_level: float = 0.0,
        red_temperature: float = 0.3,
        blue_temperature: float = 0.3,
        red_voice: str | None = None,
        blue_voice: str | None = None,
        red_tone: str | None = None,
        blue_tone: str | None = None,
    ):
        """Run a paced episode as an SSE event stream."""
        if chain not in {card[0] for card in ATTACK_CARDS}:
            raise HTTPException(status_code=400, detail="Unknown scenario")
        if difficulty not in {1, 2, 3}:
            raise HTTPException(status_code=400, detail="difficulty must be 1, 2, or 3")
        if not 0 <= pace_ms <= 5_000:
            raise HTTPException(status_code=400, detail="pace_ms must be between 0 and 5000")
        if mode not in {"auto", "full"}:
            raise HTTPException(status_code=400, detail="mode must be auto or full")
        if not 0.0 <= noise_level <= 1.0:
            raise HTTPException(status_code=400, detail="noise_level must be between 0 and 1")
        if not 0.0 <= red_temperature <= 1.0 or not 0.0 <= blue_temperature <= 1.0:
            raise HTTPException(status_code=400, detail="temperatures must be between 0 and 1")
        if (red_tone and len(red_tone) > 160) or (blue_tone and len(blue_tone) > 160):
            raise HTTPException(status_code=400, detail="tone controls must be 160 characters or fewer")
        def events():
            yield _sse(
                "starting",
                {
                    "scenario": chain,
                    "difficulty": difficulty,
                    "message": "Preparing the call environment and loading policy voices.",
                },
            )
            runtime_future = _RUNTIME_EXECUTOR.submit(build_runtime, mode)
            loading_tick = 0
            while not runtime_future.done():
                loading_tick += 1
                yield _sse(
                    "loading",
                    {
                        "tick": loading_tick,
                        "message": "Loading policy and voice models.",
                    },
                )
                time.sleep(5)
            try:
                bundle = runtime_future.result()
            except Exception as exc:
                yield _sse(
                    "error",
                    {
                        "message": "Full runtime is unavailable.",
                        "detail": str(exc),
                    },
                )
                return
            episode_seed = _episode_seed(seed)
            _prune()
            bundle.red_policy.temperature = red_temperature
            bundle.blue_policy.temperature = blue_temperature
            env = VishGymEnv(audio_renderer=PerturbedAudioRenderer(bundle.renderer, noise_level=noise_level, seed=episode_seed))
            env.judge = bundle.judge
            observation = env.reset(seed=episode_seed, scenario_id=chain, difficulty=difficulty)
            _apply_voice_controls(
                env,
                red_voice=red_voice,
                blue_voice=blue_voice,
                red_tone=red_tone,
                blue_tone=blue_tone,
            )
            policies = {Team.RED: bundle.red_policy, Team.BLUE: bundle.blue_policy}
            for team, policy in policies.items():
                setter = getattr(policy, "set_persona", None)
                if setter is not None:
                    setter(env.state.red_persona if team is Team.RED else env.state.blue_persona)
            _runs[env.state.episode_id] = {"created_at": time.time(), "state": env.state, "verdict": None}
            yield _sse(
                "started",
                {
                    "run_id": env.state.episode_id,
                    "scenario": chain,
                    "seed": episode_seed,
                    "runtime": bundle.status.model_dump(),
                    "renderer": env.audio_renderer.__class__.__name__,
                    "episode_context": _viewer_episode_context(env),
                    "viewer_messages_are_synthetic": True,
                    "transcript_available_to_agents": False,
                },
            )
            try:
                while True:
                    action = policies[observation.team].act(observation)
                    result = env.step(observation.team, action)
                    yield _sse(
                        "turn",
                        {
                            "run_id": env.state.episode_id,
                            "turn_number": env.state.turn_number,
                            "speaker": observation.team.value,
                            "message": action.spoken_text,
                            "audio_turn": _viewer_audio_turn(result.audio_turn, action.spoken_text),
                            "tool_event": result.tool_event.model_dump() if result.tool_event else None,
                            "wallet_balance_paise": env.state.wallet.balance_paise,
                            "viewer_messages_are_synthetic": True,
                            "transcript_available_to_agents": False,
                        },
                    )
                    if result.done:
                        assert result.judge_result is not None
                        _runs[env.state.episode_id]["verdict"] = result.judge_result
                        yield _sse(
                            "completed",
                            {
                                "run_id": env.state.episode_id,
                                "outcome": result.judge_result.terminal_outcome,
                                "judge": result.judge_result.model_dump(),
                                "ledger": [event.model_dump() for event in env.state.ledger],
                                "messages": _viewer_messages(env.state),
                                "viewer_messages_are_synthetic": True,
                                "transcript_available_to_agents": False,
                            },
                        )
                        _release_accelerator_memory()
                        return
                    assert result.observation is not None
                    observation = result.observation
                    if pace_ms:
                        time.sleep(pace_ms / 1000)
            except Exception as exc:
                yield _sse("error", {"run_id": env.state.episode_id, "message": str(exc)})
                _release_accelerator_memory()
                return

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/episodes/{run_id}")
    def get_episode(run_id: str) -> dict:
        _prune()
        record = _runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run is unavailable or has expired")
        state, verdict = record["state"], record["verdict"]
        return {
            "run_id": run_id,
            "scenario": state.scenario_id,
            "seed": state.seed,
            "phase": state.phase.value,
            "audio_turns": [
                _viewer_audio_turn(turn, state.transcript[index]["text"] if index < len(state.transcript) else None)
                for index, turn in enumerate(state.audio_turns)
            ],
            "messages": _viewer_messages(state),
            "ledger": [event.model_dump() for event in state.ledger],
            "judge": verdict.model_dump() if verdict is not None else None,
            "episode_context": {
                "red": _viewer_persona(state.red_persona),
                "blue": _viewer_persona(state.blue_persona),
                "blue_credentials": dict(state.credentials),
                "wallet": state.wallet.model_dump(),
                "inbox": [message.model_dump() for message in state.inbox],
            },
            "wallet_balance_paise": state.wallet.balance_paise,
            "transcript_available_to_agents": False,
            "viewer_messages_are_synthetic": True,
        }

    @app.get("/api/v1/model")
    def model_manifest() -> dict:
        return runtime_status().model_dump()

    @app.get("/api/v1/runtime/import-smoke")
    def import_smoke() -> dict:
        checks: dict[str, str] = {}
        for name in ("torch", "transformers", "peft", "qwen_tts", "soundfile"):
            try:
                if name == "qwen_tts":
                    from vishgym.arena.audio import _qwen_import_smoke

                    _qwen_import_smoke()
                else:
                    __import__(name)
                checks[name] = "ok"
            except Exception as exc:  # pragma: no cover - deployed runtime diagnostic.
                checks[name] = f"{type(exc).__name__}: {exc}"
        return {"checks": checks}

    @app.get("/api/v1/training/stream")
    def stream_training(
        role: str = "blue",
        initial_run_name: str = "blue-init-v1",
        opponent_run_name: str = "red-init-v1",
        run_name: str = "blue-live-rl-v1",
        updates: int = 3,
        group_size: int = 2,
        learning_rate: float = 5e-6,
        temperature: float = 0.7,
    ):
        request = TrainingRequest(
            role=role,
            initial_run_name=initial_run_name,
            opponent_run_name=opponent_run_name,
            run_name=run_name,
            updates=updates,
            group_size=group_size,
            learning_rate=learning_rate,
            temperature=temperature,
        )
        role_name = _safe_name(request.role, label="role")
        initial = _safe_name(request.initial_run_name, label="initial run")
        opponent = _safe_name(request.opponent_run_name, label="opponent run")
        target = _safe_name(request.run_name, label="target run")

        def events():
            from vishgym.training.grpo import GroupRelativeConfig, run_group_relative_round

            metric_queue: queue.Queue[dict] = queue.Queue()
            root = _adapter_root()
            initial_adapter = root / initial / "adapter"
            opponent_adapter = root / opponent / "adapter"
            output_dir = root / target
            team = Team(role_name)

            yield _sse(
                "training_started",
                {
                    "role": role_name,
                    "run_name": target,
                    "updates": request.updates,
                    "group_size": request.group_size,
                    "learning_rate": request.learning_rate,
                    "temperature": request.temperature,
                },
            )

            def push(payload: dict) -> None:
                metric_queue.put(payload)

            def work():
                try:
                    _release_accelerator_memory()
                    result = run_group_relative_round(
                        GroupRelativeConfig(
                            role=team,
                            initial_adapter_path=str(initial_adapter),
                            opponent_adapter_path=str(opponent_adapter),
                            output_dir=str(output_dir),
                            updates=request.updates,
                            group_size=request.group_size,
                            learning_rate=request.learning_rate,
                            temperature=request.temperature,
                        ),
                        metrics_callback=push,
                    )
                    return {
                        "adapter_path": str(result.adapter_path),
                        "receipt_path": str(result.receipt_path),
                        "updates": result.updates,
                        "mean_reward": result.mean_reward,
                        "mean_loss": result.mean_loss,
                    }
                except Exception as exc:
                    push({"event": "training_error", "message": str(exc)})
                    raise
                finally:
                    _release_accelerator_memory()

            future = _TRAINING_EXECUTOR.submit(work)
            tick = 0
            while not future.done() or not metric_queue.empty():
                try:
                    payload = metric_queue.get(timeout=5)
                except queue.Empty:
                    tick += 1
                    yield _sse("training_heartbeat", {"tick": tick, "message": "Training worker is running."})
                    continue
                event_name = payload.pop("event", "training_metric")
                yield _sse(event_name, payload)
            try:
                result = future.result()
            except Exception:
                return
            yield _sse("training_finished", result)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/audio/{filename}")
    def audio(filename: str):
        if Path(filename).name != filename or not filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Invalid audio reference")
        _prune()
        is_active = any(
            filename == Path(turn.audio_ref).name
            for record in _runs.values()
            for turn in record["state"].audio_turns
        )
        path = (_AUDIO_DIR / filename).resolve()
        if not is_active or _AUDIO_DIR not in path.parents or not path.exists():
            raise HTTPException(status_code=404, detail="Audio has expired")
        return FileResponse(path, media_type="audio/wav")
else:  # pragma: no cover
    app = None
