from __future__ import annotations

from vishgym.arena.models import EpisodeState, JudgeResult, Team
from vishgym.arena.world import VishGymEnv
from vishgym.dev.policies import DeterministicTestPolicy


def run_local_episode(
    seed: int = 7,
    scenario_id: str = "vishing_collect",
    difficulty: int = 2,
) -> tuple[EpisodeState, JudgeResult]:
    env = VishGymEnv()
    observation = env.reset(seed=seed, scenario_id=scenario_id, difficulty=difficulty)
    red = DeterministicTestPolicy(Team.RED)
    blue = DeterministicTestPolicy(Team.BLUE)
    while True:
        policy = red if observation.team is Team.RED else blue
        result = env.step(observation.team, policy.act(observation))
        if result.done:
            assert result.judge_result is not None
            return env.state, result.judge_result
        assert result.observation is not None
        observation = result.observation
