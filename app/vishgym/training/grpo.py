"""Small, closed-environment Group Relative Policy Optimization rounds.

The implementation is intentionally conservative: it samples groups of complete
episodes from one role policy, normalizes only terminal sandbox rewards within a
group, and performs one LoRA-only policy-gradient update.  It does not save raw
model completions, transcripts, or audio after the round finishes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import tempfile
from typing import Any

from vishgym.arena.audio import QwenCustomVoiceRenderer
from vishgym.arena.models import AgentAction, AgentObservation, Persona, Team, ToolCall
from vishgym.core.agents import GemmaPolicyHarness, ScriptedPolicy
from vishgym.core.fixtures import ATTACK_CARDS
from vishgym.core.prompting import policy_system_prompt, policy_user_content
from vishgym.training.sft import BASE_MODEL_ID, WarmStartConfig, _load_qlora_model, training_preflight


@dataclass(frozen=True)
class GroupRelativeConfig:
    role: Team
    initial_adapter_path: str
    output_dir: str
    model_id: str = BASE_MODEL_ID
    opponent_adapter_path: str | None = None
    scenario_ids: tuple[str, ...] = ("vishing_collect", "smishing_link", "whatsapp_beneficiary")
    seed: int = 211
    updates: int = 3
    group_size: int = 2
    learning_rate: float = 5e-6
    temperature: float = 0.7
    max_new_tokens: int = 180
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass(frozen=True)
class GroupRelativeResult:
    adapter_path: Path
    receipt_path: Path
    updates: int
    mean_reward: float
    mean_loss: float


@dataclass
class _TurnTrace:
    messages: list[dict[str, Any]]
    completion: str
    reward: float = 0.0
    advantage: float = 0.0


def _messages(team: Team, persona: Persona, observation: AgentObservation, audio_dir: Path) -> list[dict[str, Any]]:
    audio_path = None
    if observation.opponent_audio_ref is not None:
        audio_path = (audio_dir / Path(observation.opponent_audio_ref).name).resolve()
        if audio_dir not in audio_path.parents or not audio_path.is_file():
            raise ValueError("opponent audio reference is not available in this closed rollout")
    content = policy_user_content(
        turn_number=observation.turn_number,
        available_tools=observation.available_tools,
        own_tools=observation.own_tools,
        audio_path=audio_path,
    )
    return [{"role": "system", "content": policy_system_prompt(team, persona)}, {"role": "user", "content": content}]


def _parse_action(raw: str, observation: AgentObservation) -> AgentAction:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return AgentAction(
            spoken_text="I could not form a valid synthetic action.",
            tool_call=ToolCall(name="invalid.model_output", arguments={}),
        )
    try:
        payload = json.loads(match.group(0))
        tool_payload = payload.get("tool_call")
        tool_call = ToolCall.model_validate(tool_payload) if tool_payload is not None else None
        return AgentAction(spoken_text=payload["spoken_text"], tool_call=tool_call)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return AgentAction(
            spoken_text="I could not form a valid synthetic action.",
            tool_call=ToolCall(name="invalid.model_output", arguments={}),
        )


class _SamplingPolicy:
    def __init__(self, *, team: Team, model: Any, processor: Any, audio_dir: Path, temperature: float, max_new_tokens: int):
        self.team = team
        self.model = model
        self.processor = processor
        self.audio_dir = audio_dir
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.persona: Persona | None = None
        self.traces: list[_TurnTrace] = []

    def set_persona(self, persona: Persona) -> None:
        if persona.role is not self.team:
            raise ValueError("actor persona must match the trained role")
        self.persona = persona

    def act(self, observation: AgentObservation) -> AgentAction:
        import torch

        if self.persona is None:
            raise RuntimeError("synthetic actor persona was not initialized")
        messages = _messages(self.team, self.persona, observation, self.audio_dir)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=max(self.temperature, 0.01),
                top_p=0.95,
            )
        raw = self.processor.decode(output[0][input_length:], skip_special_tokens=True)
        self.traces.append(_TurnTrace(messages=messages, completion=raw))
        return _parse_action(raw, observation)


def _completion_log_probability(model: Any, processor: Any, trace: _TurnTrace):
    """Mean token log-probability for one sampled JSON completion."""
    import torch

    full_messages = [*trace.messages, {"role": "assistant", "content": trace.completion}]
    prompt = processor.apply_chat_template(
        trace.messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    full = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
        enable_thinking=False,
    ).to(model.device)
    prompt_length = prompt["input_ids"].shape[-1]
    ids = full["input_ids"]
    if ids.shape[-1] <= prompt_length or not torch.equal(ids[:, :prompt_length], prompt["input_ids"]):
        raise ValueError("Gemma chat template did not preserve the rollout audio prompt prefix")
    labels = ids.clone()
    labels[:, :prompt_length] = -100
    logits = model(**full).logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target.ne(-100)
    if not bool(mask.any()):
        raise ValueError("sampled action contained no trainable completion tokens")
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(-1, target.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return token_log_probs.masked_select(mask).mean()


def _opponent_policy(config: GroupRelativeConfig, team: Team, audio_dir: Path):
    if config.opponent_adapter_path is None:
        return ScriptedPolicy(team)
    opponent = GemmaPolicyHarness(team=team, adapter_path=config.opponent_adapter_path, model_id=config.model_id, audio_dir=audio_dir)
    opponent.load()
    return opponent


def run_group_relative_round(config: GroupRelativeConfig) -> GroupRelativeResult:
    """Run a small online, no-upload GRPO-style LoRA round in the closed arena.

    Groups share a scenario/seed and are normalized by terminal reward. A zero-KL
    objective is intentional for this compact first round; every reward still comes
    from the fixed sandbox judge. Historical adapters can be supplied as frozen
    opponents, while the initial experiment may use the reviewed scripted policy.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install vishgym[training] before running group-relative optimization.") from exc
    if config.role not in {Team.RED, Team.BLUE}:
        raise ValueError("group-relative optimization supports red or blue roles only")
    if config.group_size < 2:
        raise ValueError("group_size must be at least 2 for relative advantages")
    if config.updates < 1:
        raise ValueError("updates must be positive")
    known = {scenario_id for scenario_id, _ in ATTACK_CARDS}
    if not config.scenario_ids or set(config.scenario_ids) - known:
        raise ValueError("GRPO scenarios must be configured synthetic attack cards")
    if not Path(config.initial_adapter_path).is_dir():
        raise ValueError("initial_adapter_path must point to a reviewed local warm-start adapter")
    training_preflight()
    torch.manual_seed(config.seed)
    model, processor = _load_qlora_model(
        WarmStartConfig(
            dataset_root="unused-by-grpo",
            output_dir=config.output_dir,
            role=config.role,
            model_id=config.model_id,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            initial_adapter_path=config.initial_adapter_path,
        )
    )
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rewards: list[float] = []
    losses: list[float] = []

    # Qwen output is used only while an active rollout needs it. The temporary
    # directory is deleted at the end of this function, along with raw completion
    # traces, satisfying the ephemeral synthetic-data policy.
    with tempfile.TemporaryDirectory(prefix="vishgym-grpo-audio-") as temporary:
        audio_dir = Path(temporary)
        renderer = QwenCustomVoiceRenderer(output_dir=audio_dir)
        renderer.load()
        opponent_team = Team.BLUE if config.role is Team.RED else Team.RED
        opponent = _opponent_policy(config, opponent_team, audio_dir)
        for update in range(config.updates):
            scenario_id = config.scenario_ids[update % len(config.scenario_ids)]
            group: list[tuple[_SamplingPolicy, float]] = []
            for sample_index in range(config.group_size):
                actor = _SamplingPolicy(
                    team=config.role,
                    model=model,
                    processor=processor,
                    audio_dir=audio_dir,
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                )
                # GRPO compares sampled completions for the same environment setup.
                # Stochastic generation, rather than a changed persona/tool state,
                # supplies the within-group variation.
                seed = config.seed + update
                if config.role is Team.RED:
                    from vishgym.training.rollouts import run_policy_episode

                    rollout = run_policy_episode(
                        red_policy=actor,
                        blue_policy=opponent,
                        seed=seed,
                        scenario_id=scenario_id,
                        audio_renderer=renderer,
                    )
                    reward = rollout.verdict.red_reward
                else:
                    from vishgym.training.rollouts import run_policy_episode

                    rollout = run_policy_episode(
                        red_policy=opponent,
                        blue_policy=actor,
                        seed=seed,
                        scenario_id=scenario_id,
                        audio_renderer=renderer,
                    )
                    reward = rollout.verdict.blue_reward
                for trace in actor.traces:
                    trace.reward = reward
                group.append((actor, reward))
                rewards.append(reward)
            values = torch.tensor([reward for _, reward in group], device=model.device, dtype=torch.float32)
            advantages = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
            weighted_losses = []
            model.train()
            for (actor, _), advantage in zip(group, advantages, strict=True):
                for trace in actor.traces:
                    trace.advantage = float(advantage.item())
                    weighted_losses.append(-advantage * _completion_log_probability(model, processor, trace))
            if not weighted_losses:
                continue
            loss = torch.stack(weighted_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    receipt = {
        "schema_version": "vishgym-group-relative-receipt/v1",
        "role": config.role.value,
        "base_model": config.model_id,
        "initial_adapter_path": str(Path(config.initial_adapter_path).resolve()),
        "opponent_adapter_path": str(Path(config.opponent_adapter_path).resolve()) if config.opponent_adapter_path else "scripted-reviewed-baseline",
        "config": {**asdict(config), "role": config.role.value, "scenario_ids": list(config.scenario_ids)},
        "updates_completed": len(losses),
        "mean_terminal_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "mean_policy_loss": round(sum(losses) / len(losses), 6) if losses else 0.0,
        "synthetic_only": True,
        "raw_completions_persisted": False,
        "automatic_publication": False,
        "promotion_status": "review_required",
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return GroupRelativeResult(
        adapter_path=adapter_path,
        receipt_path=receipt_path,
        updates=len(losses),
        mean_reward=float(receipt["mean_terminal_reward"]),
        mean_loss=float(receipt["mean_policy_loss"]),
    )
