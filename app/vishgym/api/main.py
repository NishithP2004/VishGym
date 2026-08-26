from __future__ import annotations

import time
from pathlib import Path

from vishgym.arena.runner import run_local_episode
from vishgym.core.fixtures import ATTACK_CARDS

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - handled at deployment time
    FastAPI = None


if FastAPI is not None:
    app = FastAPI(title="VishGym", version="0.1.0", description="Closed synthetic red/blue self-play arena")
    _runs: dict[str, dict] = {}
    _RUN_TTL_SECONDS = 60 * 60
    _AUDIO_DIR = Path("artifacts/runtime/audio").resolve()

    class SimulationRequest(BaseModel):
        chain: str = Field(default="vishing_collect")
        difficulty: int = Field(default=2, ge=1, le=3)
        seed: int = Field(default=7, ge=0)

    def _prune() -> None:
        cutoff = time.time() - _RUN_TTL_SECONDS
        for run_id in [key for key, value in _runs.items() if value["created_at"] < cutoff]:
            state = _runs[run_id]["state"]
            for turn in state.audio_turns:
                path = (_AUDIO_DIR / Path(turn.audio_ref).name).resolve()
                if _AUDIO_DIR in path.parents and path.exists():
                    path.unlink()
            del _runs[run_id]

    @app.get("/api/v1/catalogue")
    def catalogue() -> dict:
        return {"cards": [{"id": card_id, "title": title} for card_id, title in ATTACK_CARDS], "synthetic_only": True}

    @app.post("/api/v1/simulations")
    def create_simulation(request: SimulationRequest) -> dict:
        if request.chain not in {card[0] for card in ATTACK_CARDS}:
            raise HTTPException(status_code=400, detail="Unknown synthetic scenario")
        _prune()
        state, verdict = run_local_episode(
            seed=request.seed,
            scenario_id=request.chain,
            difficulty=request.difficulty,
        )
        _runs[state.episode_id] = {"created_at": time.time(), "state": state, "verdict": verdict}
        return {"run_id": state.episode_id, "status": "complete", "expires_in_seconds": _RUN_TTL_SECONDS}

    @app.get("/api/v1/simulations/{run_id}")
    def get_simulation(run_id: str) -> dict:
        _prune()
        record = _runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run is unavailable or has expired")
        state, verdict = record["state"], record["verdict"]
        return {
            "run_id": run_id,
            "scenario": state.scenario_id,
            "phase": state.phase.value,
            "audio_turns": [turn.model_dump() for turn in state.audio_turns],
            "ledger": [event.model_dump() for event in state.ledger],
            "judge": verdict.model_dump(),
            "wallet_balance_paise": state.wallet.balance_paise,
            "transcript_available_to_agents": False,
        }

    @app.get("/api/v1/model")
    def model_manifest() -> dict:
        return {
            "base_model": "google/gemma-4-E2B-it",
            "tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "runtime": "scripted local fallback until reviewed QLoRA adapters are supplied",
            "synthetic_only": True,
        }

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
