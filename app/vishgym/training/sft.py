"""GPU-only QLoRA supervised warm-start for one VishGym role adapter.

This module is deliberately separate from the deterministic demo.  Imports for
Torch, Transformers, and PEFT are lazy so a local safety/test run never downloads
weights or requires a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from typing import Any

from vishgym.arena.models import Team
from vishgym.core.prompting import policy_system_prompt, policy_user_content
from vishgym.training.dataset import TrainingExample, load_training_examples, require_trainable_audio_dataset


BASE_MODEL_ID = "google/gemma-4-E2B-it"


@dataclass(frozen=True)
class WarmStartConfig:
    dataset_root: str
    output_dir: str
    role: Team
    model_id: str = BASE_MODEL_ID
    max_steps: int = 60
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 7
    save_steps: int = 30
    initial_adapter_path: str | None = None


@dataclass(frozen=True)
class WarmStartResult:
    adapter_path: Path
    receipt_path: Path
    dataset_revision: str
    role: str
    examples: int
    metrics: dict[str, float]


def training_preflight(*, require_cuda: bool = True) -> dict[str, Any]:
    """Inspect local runtime prerequisites without downloading a model or token."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install vishgym[training] in the Colab runtime first.") from exc
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("A CUDA GPU runtime is required for the 4-bit Gemma QLoRA training stage.")
    return {
        "python": platform.python_version(),
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else False,
        "hf_token_configured": bool(__import__("os").environ.get("HF_TOKEN")),
        "base_model": BASE_MODEL_ID,
        "network_download_started": False,
    }


def action_target(example: TrainingExample) -> str:
    """Serialize the only output that may be supervised for this decision."""
    return json.dumps(
        {
            "spoken_text": example.target_spoken_text,
            "tool_call": (
                None
                if example.target_tool_name is None
                else {"name": example.target_tool_name, "arguments": example.target_tool_arguments}
            ),
        },
        sort_keys=True,
    )


def policy_messages(example: TrainingExample, dataset_root: str | Path) -> list[dict[str, Any]]:
    """Build an audio-first prompt without adding an opponent transcript."""
    audio_path = None
    if example.opponent_audio is not None:
        root = Path(dataset_root).resolve()
        audio_path = (root / example.opponent_audio.path).resolve()
        if root not in audio_path.parents or not audio_path.is_file():
            raise ValueError(f"opponent audio is not a local artifact for {example.example_id}")
    content = policy_user_content(
        turn_number=example.turn_number,
        available_tools=example.available_tools,
        own_tools=example.own_tools,
        audio_path=audio_path,
    )
    return [
        {"role": "system", "content": policy_system_prompt(Team(example.team), example.persona)},
        {"role": "user", "content": content},
        {"role": "assistant", "content": action_target(example)},
    ]


class _AudioPromptDataset:
    """Lazy one-example batches keep audio feature padding architecture-agnostic."""

    def __init__(self, examples: list[TrainingExample], dataset_root: Path, processor: Any):
        self.examples = examples
        self.dataset_root = dataset_root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        messages = policy_messages(self.examples[index], self.dataset_root)
        prompt_messages = messages[:-1]
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = prompt["input_ids"]
        full_ids = full["input_ids"]
        prompt_length = prompt_ids.shape[-1]
        if full_ids.shape[-1] <= prompt_length or not torch.equal(full_ids[:, :prompt_length], prompt_ids):
            raise ValueError("Gemma chat template did not preserve the audio prompt prefix")
        labels = full_ids.clone()
        labels[:, :prompt_length] = -100
        full["labels"] = labels
        return dict(full)


def _single_audio_collator(features: list[dict[str, Any]]) -> dict[str, Any]:
    if len(features) != 1:
        raise ValueError("VishGym audio warm-start uses per-device batch size 1; increase gradient accumulation instead")
    return features[0]


def _load_qlora_model(config: WarmStartConfig) -> tuple[Any, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install vishgym[training] before starting a QLoRA warm-start run.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA warm-start requires CUDA; use a Colab GPU runtime.")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(config.model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        config.model_id,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    requested = ("q_proj", "k_proj", "v_proj", "o_proj")
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [name for name in requested if name in available]
    if len(targets) != len(requested):
        raise RuntimeError(f"Gemma projection modules unavailable for LoRA: expected {requested}, found {sorted(available)[:20]}")
    if config.initial_adapter_path:
        return PeftModel.from_pretrained(model, config.initial_adapter_path, is_trainable=True), processor
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_config), processor


def run_warm_start(config: WarmStartConfig) -> WarmStartResult:
    """Train and save one local adapter. This function never uploads or promotes it."""
    try:
        import torch
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("Install vishgym[training] before starting a QLoRA warm-start run.") from exc
    preflight = training_preflight()
    manifest = require_trainable_audio_dataset(config.dataset_root)
    examples = load_training_examples(config.dataset_root, team=config.role)
    if not examples:
        raise ValueError(f"no {config.role.value} examples found in the dataset")
    set_seed(config.seed)
    model, processor = _load_qlora_model(config)
    dataset = _AudioPromptDataset(examples, Path(config.dataset_root).resolve(), processor)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        bf16=bool(preflight["bf16_supported"]),
        fp16=not bool(preflight["bf16_supported"]),
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, config.save_steps),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=_single_audio_collator,
    )
    train_output = trainer.train()
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    metrics = {key: float(value) for key, value in train_output.metrics.items() if isinstance(value, int | float)}
    receipt = {
        "schema_version": "vishgym-warm-start-receipt/v1",
        "role": config.role.value,
        "base_model": config.model_id,
        "dataset_revision": manifest["revision"],
        "examples": len(examples),
        "config": {**asdict(config), "role": config.role.value},
        "metrics": metrics,
        "synthetic_only": True,
        "automatic_publication": False,
        "promotion_status": "review_required",
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return WarmStartResult(
        adapter_path=adapter_path,
        receipt_path=receipt_path,
        dataset_revision=manifest["revision"],
        role=config.role.value,
        examples=len(examples),
        metrics=metrics,
    )
