"""Synthetic warm-start traces that retain no external identities or endpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from vishgym.arena.models import Team
from vishgym.arena.runner import run_local_episode
from vishgym.arena.world import VishGymEnv
from vishgym.core.fixtures import ATTACK_CARDS


@dataclass(frozen=True)
class WarmStartExample:
    scenario_id: str
    seed: int
    team: str
    audio_ref: str | None
    available_tools: list[str]
    target_spoken_text: str
    target_tool_name: str | None
    target_tool_arguments: dict
    synthetic_only: bool = True

    def model_dump(self) -> dict:
        return asdict(self)


def build_warm_start_examples(
    seeds: Iterable[int] = (7, 11),
    scenario_ids: Iterable[str] | None = None,
) -> list[WarmStartExample]:
    """Build reviewed local rollouts as audio-first role-action supervision.

    The target text is an output label, never part of the opponent observation at
    runtime. The function is intentionally deterministic for a fixed seed list.
    """
    card_ids = [scenario_id for scenario_id, _ in ATTACK_CARDS]
    known = set(card_ids)
    selected = card_ids if scenario_ids is None else list(scenario_ids)
    unknown = set(selected) - known
    if unknown:
        raise ValueError(f"warm-start data accepts synthetic attack cards only: {sorted(unknown)}")
    examples: list[WarmStartExample] = []
    for scenario_id in selected:
        for seed in seeds:
            state, _ = run_local_episode(seed=seed, scenario_id=scenario_id)
            for index, transcript_turn in enumerate(state.transcript):
                event = state.ledger[index] if index < len(state.ledger) else None
                examples.append(
                    WarmStartExample(
                        scenario_id=scenario_id,
                        seed=seed,
                        team=transcript_turn["speaker"],
                        audio_ref=state.audio_turns[index - 1].audio_ref if index else None,
                        available_tools=sorted(
                            VishGymEnv.RED_TOOLS if transcript_turn["speaker"] == Team.RED.value else VishGymEnv.BLUE_TOOLS
                        ),
                        target_spoken_text=transcript_turn["text"],
                        target_tool_name=event.tool if event else None,
                        target_tool_arguments=event.arguments if event else {},
                    )
                )
    return examples
