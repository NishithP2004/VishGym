# Alternating QLoRA/GRPO Training

## Prerequisites

1. Configure the Google Colab MCP bridge locally and open `notebooks/VishGym_Colab.ipynb` in an authenticated Colab session.
2. Accept the Gemma model terms and supply scoped, private Hugging Face credentials through the Colab secret manager.
3. Keep all generated rollouts in a reviewed synthetic artifact repository.

## Promotion cycle

1. Start from the fixed hybrid judge and a frozen Blue adapter.
2. Train Red QLoRA with GRPO on safe synthetic rollouts.
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
