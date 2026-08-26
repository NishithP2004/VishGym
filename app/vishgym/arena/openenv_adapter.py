"""Typed OpenEnv server adapter for the closed VishGym world.

This module is imported only in the ``training`` extra, so the deterministic
local demo remains lightweight. It exposes redacted state and observations:
the episode transcript is intentionally never serialised through OpenEnv.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server import create_fastapi_app

try:  # OpenEnv moved these public types in the 0.4 series.
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:  # pragma: no cover - exercised only with earlier OpenEnv builds
    from openenv.core.env_server import Action, Environment, Observation, State

from vishgym.arena.models import AgentAction, AgentObservation, EpisodeState, Team, ToolCall
from vishgym.arena.world import VishGymEnv


class VishGymAction(Action):
    team: Team
    spoken_text: str
    tool_call: ToolCall | None = None


class VishGymObservation(Observation):
    episode_id: str
    team: Team
    turn_number: int
    phase: str
    opponent_audio_ref: str | None
    own_tools: dict[str, Any]
    available_tools: list[str]
    last_tool_event: dict[str, Any] | None = None
    terminal_outcome: str | None = None


class VishGymState(State):
    scenario_id: str = ""
    phase: str = "setup"
    difficulty: int = 1
    ledger_count: int = 0
    audio_turn_count: int = 0
    transcript_retained_for_judge_only: bool = True


class OpenEnvVishGymEnvironment(Environment[VishGymAction, VishGymObservation, VishGymState]):
    """Server-side OpenEnv wrapper with delayed, per-team terminal reward."""

    REQUIRES_SINGLE_THREAD_EXECUTOR = True

    def __init__(self) -> None:
        super().__init__()
        self._world = VishGymEnv()
        self._last_reward = 0.0

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario_id: str = "vishing_collect",
        difficulty: int = 2,
        **_: Any,
    ) -> VishGymObservation:
        del episode_id  # VishGym always issues an opaque fresh episode identifier.
        observation = self._world.reset(seed=7 if seed is None else seed, scenario_id=scenario_id, difficulty=difficulty)
        self._last_reward = 0.0
        return self._to_openenv_observation(observation)

    def step(self, action: VishGymAction, timeout_s: float | None = None, **_: Any) -> VishGymObservation:
        del timeout_s
        result = self._world.step(
            action.team,
            AgentAction(spoken_text=action.spoken_text, tool_call=action.tool_call),
        )
        if result.done:
            assert result.judge_result is not None
            self._last_reward = (
                result.judge_result.red_reward if action.team is Team.RED else result.judge_result.blue_reward
            )
            return VishGymObservation(
                **self._redacted_observation_fields(action.team),
                last_tool_event=result.tool_event.model_dump() if result.tool_event else None,
                terminal_outcome=result.judge_result.terminal_outcome,
                reward=self._last_reward,
                done=True,
            )
        assert result.observation is not None
        self._last_reward = 0.0
        return self._to_openenv_observation(result.observation, result.tool_event.model_dump() if result.tool_event else None)

    @property
    def state(self) -> VishGymState:
        state = self._world.state
        return VishGymState(
            episode_id=state.episode_id,
            step_count=state.turn_number,
            scenario_id=state.scenario_id,
            phase=state.phase.value,
            difficulty=state.difficulty,
            ledger_count=len(state.ledger),
            audio_turn_count=len(state.audio_turns),
        )

    def _to_openenv_observation(
        self,
        observation: AgentObservation,
        last_tool_event: dict[str, Any] | None = None,
    ) -> VishGymObservation:
        return VishGymObservation(
            episode_id=observation.episode_id,
            team=observation.team,
            turn_number=observation.turn_number,
            phase=observation.phase.value,
            opponent_audio_ref=observation.opponent_audio_ref,
            own_tools=observation.own_tools,
            available_tools=observation.available_tools,
            last_tool_event=last_tool_event,
            reward=0.0,
            done=False,
        )

    def _redacted_observation_fields(self, team: Team) -> dict[str, Any]:
        state: EpisodeState = self._world.state
        return {
            "episode_id": state.episode_id,
            "team": team,
            "turn_number": state.turn_number,
            "phase": state.phase.value,
            "opponent_audio_ref": state.audio_turns[-1].audio_ref if state.audio_turns else None,
            "own_tools": {
                "wallet_balance_paise": state.wallet.balance_paise,
                "inbox_count": len(state.inbox),
                "portal_count": len(state.portals),
                "scenario_id": state.scenario_id,
            },
            "available_tools": sorted(self._world.RED_TOOLS if team is Team.RED else self._world.BLUE_TOOLS),
        }


def create_environment() -> OpenEnvVishGymEnvironment:
    return _SERVER_ENV


_SERVER_ENV = OpenEnvVishGymEnvironment()
app = create_fastapi_app(create_environment, VishGymAction, VishGymObservation, max_concurrent_envs=1)
