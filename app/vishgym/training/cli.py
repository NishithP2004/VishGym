"""Command-line entry points for reviewed, local-only VishGym training stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from vishgym.arena.audio import QwenCustomVoiceRenderer
from vishgym.arena.models import Team
from vishgym.core.agents import GemmaPolicyHarness
from vishgym.training.dataset import export_warm_start_dataset
from vishgym.training.evaluation import write_review_manifest
from vishgym.training.grpo import GroupRelativeConfig, run_group_relative_round
from vishgym.training.rollouts import (
    evaluate_blue_policy,
    evaluate_red_policy,
    write_evaluation,
    write_red_review_manifest,
)
from vishgym.training.sft import WarmStartConfig, run_warm_start, training_preflight


def _team(value: str) -> Team:
    try:
        team = Team(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("role must be red or blue") from exc
    if team not in {Team.RED, Team.BLUE}:
        raise argparse.ArgumentTypeError("role must be red or blue")
    return team


def _print(payload: object) -> None:
    if hasattr(payload, "__dict__"):
        payload = {key: str(value) if isinstance(value, Path) else value for key, value in payload.__dict__.items()}
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _export(args: argparse.Namespace) -> None:
    root = Path(args.output_dir).resolve()
    if args.renderer == "qwen":
        with tempfile.TemporaryDirectory(prefix="vishgym-qwen-export-") as temporary:
            renderer = QwenCustomVoiceRenderer(output_dir=temporary)
            renderer.load()
            result = export_warm_start_dataset(
                root,
                seeds=args.seeds,
                scenario_ids=args.scenarios,
                renderer=renderer,
                difficulty=args.difficulty,
            )
    else:
        result = export_warm_start_dataset(root, seeds=args.seeds, scenario_ids=args.scenarios, difficulty=args.difficulty)
    _print({
        "root": result.root,
        "examples": result.example_count,
        "revision": result.revision,
        "audio_training_eligible": result.audio_training_eligible,
        "automatic_publication": False,
    })


def _warm_start(args: argparse.Namespace) -> None:
    result = run_warm_start(
        WarmStartConfig(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            role=args.role,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            save_steps=args.save_steps,
            initial_adapter_path=args.initial_adapter_path,
        )
    )
    _print(result)


def _grpo(args: argparse.Namespace) -> None:
    result = run_group_relative_round(
        GroupRelativeConfig(
            role=args.role,
            initial_adapter_path=args.initial_adapter_path,
            output_dir=args.output_dir,
            opponent_adapter_path=args.opponent_adapter_path,
            scenario_ids=tuple(args.scenarios),
            seed=args.seed,
            updates=args.updates,
            group_size=args.group_size,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    )
    _print(result)


def _evaluate(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="vishgym-evaluation-") as temporary:
        audio_dir = Path(temporary)
        renderer = QwenCustomVoiceRenderer(output_dir=audio_dir)
        renderer.load()
        blue = GemmaPolicyHarness(team=Team.BLUE, adapter_path=args.blue_adapter_path, audio_dir=audio_dir)
        blue.load()
        red = None
        if args.red_adapter_path:
            red = GemmaPolicyHarness(team=Team.RED, adapter_path=args.red_adapter_path, audio_dir=audio_dir)
            red.load()
        evaluation = evaluate_blue_policy(
            blue_policy=blue,
            red_policy=red,
            seeds=args.seeds,
            fraud_scenarios=args.scenarios,
            audio_renderer=renderer,
            dataset_revision=args.dataset_revision,
            adapter_revision=args.adapter_revision,
        )
    evaluation_path = write_evaluation(args.output_path, evaluation)
    manifest_path = write_review_manifest(
        Path(args.output_path).with_name(f"{args.adapter_revision}.review.json"),
        role="blue",
        candidate_adapter=args.blue_adapter_path,
        report=evaluation.report,
        reviewer=args.reviewer,
    )
    _print({"evaluation_path": evaluation_path, "review_manifest": manifest_path, **evaluation.model_dump()})


def _evaluate_red(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="vishgym-red-evaluation-") as temporary:
        audio_dir = Path(temporary)
        renderer = QwenCustomVoiceRenderer(output_dir=audio_dir)
        renderer.load()
        red = GemmaPolicyHarness(team=Team.RED, adapter_path=args.red_adapter_path, audio_dir=audio_dir)
        red.load()
        blue = None
        if args.blue_adapter_path:
            blue = GemmaPolicyHarness(team=Team.BLUE, adapter_path=args.blue_adapter_path, audio_dir=audio_dir)
            blue.load()
        evaluation = evaluate_red_policy(
            red_policy=red,
            blue_policy=blue,
            seeds=args.seeds,
            fraud_scenarios=args.scenarios,
            audio_renderer=renderer,
            adapter_revision=args.adapter_revision,
            opponent_revision=args.opponent_revision,
        )
    evaluation_path = Path(args.output_path)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = write_red_review_manifest(
        evaluation_path.with_name(f"{args.adapter_revision}.review.json"),
        candidate_adapter=args.red_adapter_path,
        evaluation=evaluation,
        reviewer=args.reviewer,
    )
    _print({"evaluation_path": evaluation_path, "review_manifest": manifest_path, **evaluation.model_dump()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed, synthetic-only VishGym training tools")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("preflight", help="verify local GPU prerequisites without downloading a model")

    export = subcommands.add_parser("export-dataset", help="export content-addressed synthetic WAVs and warm-start labels")
    export.add_argument("--output-dir", required=True)
    export.add_argument("--renderer", choices=("qwen", "synthetic-test"), default="synthetic-test")
    export.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    export.add_argument("--scenarios", nargs="+", default=None)
    export.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2)

    warm = subcommands.add_parser("warm-start", help="train one local QLoRA role adapter from a Qwen-audio export")
    warm.add_argument("--dataset-root", required=True)
    warm.add_argument("--output-dir", required=True)
    warm.add_argument("--role", type=_team, required=True)
    warm.add_argument("--max-steps", type=int, default=60)
    warm.add_argument("--learning-rate", type=float, default=1e-5)
    warm.add_argument("--gradient-accumulation-steps", type=int, default=8)
    warm.add_argument("--save-steps", type=int, default=30)
    warm.add_argument("--seed", type=int, default=7)
    warm.add_argument("--initial-adapter-path")

    grpo = subcommands.add_parser("grpo", help="run one closed, group-relative self-play LoRA round")
    grpo.add_argument("--role", type=_team, required=True)
    grpo.add_argument("--initial-adapter-path", required=True)
    grpo.add_argument("--output-dir", required=True)
    grpo.add_argument("--opponent-adapter-path")
    grpo.add_argument("--scenarios", nargs="+", default=["vishing_collect", "smishing_link", "whatsapp_beneficiary"])
    grpo.add_argument("--updates", type=int, default=3)
    grpo.add_argument("--group-size", type=int, default=2)
    grpo.add_argument("--learning-rate", type=float, default=5e-6)
    grpo.add_argument("--temperature", type=float, default=0.7)
    grpo.add_argument("--max-new-tokens", type=int, default=180)
    grpo.add_argument("--seed", type=int, default=211)

    evaluate = subcommands.add_parser("evaluate-blue", help="evaluate a Blue adapter and write a review-only manifest")
    evaluate.add_argument("--blue-adapter-path", required=True)
    evaluate.add_argument("--red-adapter-path")
    evaluate.add_argument("--dataset-revision", required=True)
    evaluate.add_argument("--adapter-revision", required=True)
    evaluate.add_argument("--output-path", required=True)
    evaluate.add_argument("--reviewer")
    evaluate.add_argument("--seeds", type=int, nargs="+", default=[101, 103])
    evaluate.add_argument("--scenarios", nargs="+", default=None)

    evaluate_red = subcommands.add_parser("evaluate-red", help="evaluate a Red adapter and write a review-only manifest")
    evaluate_red.add_argument("--red-adapter-path", required=True)
    evaluate_red.add_argument("--blue-adapter-path")
    evaluate_red.add_argument("--adapter-revision", required=True)
    evaluate_red.add_argument("--opponent-revision", default="scripted-reviewed-baseline")
    evaluate_red.add_argument("--output-path", required=True)
    evaluate_red.add_argument("--reviewer")
    evaluate_red.add_argument("--seeds", type=int, nargs="+", default=[101, 103])
    evaluate_red.add_argument("--scenarios", nargs="+", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preflight":
        _print(training_preflight())
    elif args.command == "export-dataset":
        _export(args)
    elif args.command == "warm-start":
        _warm_start(args)
    elif args.command == "grpo":
        _grpo(args)
    elif args.command == "evaluate-blue":
        _evaluate(args)
    elif args.command == "evaluate-red":
        _evaluate_red(args)
    else:  # pragma: no cover - argparse makes this unreachable.
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
