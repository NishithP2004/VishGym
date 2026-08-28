# Alternating QLoRA/GRPO Training

## Prerequisites

1. Configure the Google Colab MCP bridge locally and open `notebooks/VishGym_Colab.ipynb` in an authenticated Colab session.
2. Accept the Gemma model terms and supply scoped, private Hugging Face credentials through the Colab secret manager.
3. Keep all generated rollouts in a reviewed synthetic artifact repository.
4. Use `vishgym-train preflight` before model download. The command does not contact Hugging Face or expose credentials.

## Runnable Colab sequence

1. Run `vishgym-train export-dataset --renderer qwen --output-dir artifacts/datasets/warm-start-qwen-v1 --seeds 7 11`.
   The export contains content-addressed local WAVs, checksums, renderer metadata, tool observations, and target actions. It contains no opponent transcript. A deterministic-tone export is test-only and is rejected by `warm-start`.
2. Train one adapter per role with `vishgym-train warm-start`. The runner uses 4-bit NF4, `prepare_model_for_kbit_training`, and LoRA on the attention projection modules. It saves an adapter and receipt locally.
3. Run `vishgym-train grpo` for Red against a scripted or immutable historical Blue opponent. A group shares the same seed/scenario; advantages are normalized from terminal sandbox rewards. Raw completions and temporary rollout audio are deleted when the command exits.
4. Run `vishgym-train evaluate-red`, review the receipt/manifest, and freeze the Red adapter path. Then run the Blue `grpo` round with that path as `--opponent-adapter-path`.
5. Run `vishgym-train evaluate-blue` to write measured F1, legitimate false-block rate, tool validity, and a review-only Blue manifest. It cannot promote or publish a model.

The bundled notebook executes this exact first round. Repeat the Red → review → Blue sequence twice more with held-out seeds, personas, timbres, and scenario combinations.

## Promotion cycle

1. Start from the fixed hybrid judge and a frozen Blue adapter.
2. Train Red QLoRA warm-start followed by group-relative updates on safe synthetic rollouts.
3. Evaluate Red against held-out personas and historical Blue adapters.
4. Review and freeze the candidate Red adapter.
5. Train Blue against that frozen Red adapter.
6. Promote Blue only when fraud-decision F1 is at least 0.80 and legitimate false-block rate is at most 10%.

## Runtime wiring

- Use `SharedGemmaAdapterRuntime` to load the shared 4-bit base once, then bind distinct reviewed Red, Blue, and fixed Judge adapters. Adapter switching is serialised because PEFT's selected adapter is mutable state.
- Attach `QwenCustomVoiceRenderer` only after loading the reviewed CustomVoice model. It accepts only the built-in speaker identifier and a fixture-controlled style instruction; it has no reference-audio parameter.
- Keep model weights in the private runtime cache. Start a closed roll-out container with `--network none` after the required model artifacts are present locally.
- The OpenEnv state/observation API is redacted. The transcript and raw tool ledger are only supplied to `FrozenGemmaContextualJudge` after an episode is terminal.

## Required evaluation logs

- base/adapters and immutable dataset revision;
- random seeds and perturbation schedule;
- valid/invalid tool-call rate;
- Blue F1, false-block rate, and simulated compromise rate;
- held-out scenario, timbre, and persona results;
- reviewer identity and promotion decision.
