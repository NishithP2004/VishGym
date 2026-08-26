from __future__ import annotations

import json

from vishgym.arena.runner import run_local_episode


def main() -> None:
    state, verdict = run_local_episode()
    print(json.dumps({"episode_id": state.episode_id, "outcome": verdict.terminal_outcome, "red_reward": verdict.red_reward, "blue_reward": verdict.blue_reward, "labels": verdict.labels}, indent=2))


if __name__ == "__main__":
    main()
